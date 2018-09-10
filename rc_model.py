# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
机器学习模型组织的经典代码
"""

import os
import time
import logging
import json
import numpy as np
import tensorflow as tf
# import sys

from tensorflow2.layers.basic_rnn import rnn
from tensorflow2.layers.match_layer import MatchLSTMLayer
from tensorflow2.layers.match_layer import AttentionFlowMatchLayer
from tensorflow2.layers.pointer_net import PointerNetDecoder

# file_dir = os.path.dirname(os.path.abspath(__file__))
# utils_dir = os.path.dirname(file_dir)
# sys.path.append(utils_dir)
# we need a base_dir

from utils import compute_bleu_rouge
from utils import normalize


class RCModel:
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, vocab, args):

        # logging
        logger = logging.getLogger("work3.rc_model")
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
        if args.log_path:
            file_handler = logging.FileHandler(args.log_path)
            # 会通过命令行传进来args.log_path，或者用默认值
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        else:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        self.logger = logger

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        # 一种选择？
        self.use_dropout = args.dropout_keep_prob < 1

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len
        # answer length
        # 一个问题，可能有多个passage（多个document的最相关的段落），可能有多个回答

        # the vocab
        self.vocab = vocab
        # 使用了组合

        # session info
        # sess_config = tf.ConfigProto(log_device_placement=True)
        sess_config = tf.ConfigProto()
        # To find out which devices your operations and tensors
        # are assigned to,
        # create the session with log_device_placement configuration
        # option set to True.
        sess_config.gpu_options.allow_growth = True
        # 并不是把gpu一开始就全部占据，而是逐步增加占掉的显存和计算力
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # gradients = self.optimizer.compute_gradients(self.loss,
        #                                              self.all_params)
        # print('gradients in _train_epoch in rc_model.py:', gradients)

        # self.capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for
        #                          grad, var in gradients if grad is not None]
        # print('capped_gradients in _train_epoch in rc_model.py:',
        #       len(capped_gradients))
        # print('capped_gradients[1] in _train_epoch in rc_model.py:',
        #       capped_gradients[1][0])
        # train_op = self.optimizer.apply_gradients(self.capped_gradients)

        # save info
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())
        # 在其他的sess.run()之前，需要先initialize the model

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._fuse()
        self._decode()
        self._compute_loss()
        self._create_train_op()
        self.logger.info(
            'Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v)))
                         for v in self.all_params])
        # 列表中数字的连乘，先乘后加，计算参数个数
        # parameter number of the model
        self.logger.info(
            'There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders, the variables that are not trainable
        一个变量，但不是在训练中会改变的那种，而且，在最后的计算中是要作为系数feed
        过去的
        """
        self.p = tf.placeholder(tf.int32, [None, None])
        # 表示是二维数据，但具体维数可以不在这里给出，具体给数据时才确定
        # 第一个维度一般是batch size
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        # 一维数据
        self.q_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        # 从实际回答到文中标注出来，比如用正则表达式，回答在文档中的什么位置，格式和实体
        # 抽取所用的格式是类似的
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        # 零维数据，也就是标量数据

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        只要词是一样的，embedding就是一样的
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            # 此处指定使用cpu
            # 第一次建立这个variable_scope, 而不是reuse
            # reuse中最重要的是模型中的trainable variable的复用
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                # 把已经初始化好的embedding传过来
                trainable=True
            )
            # 生成variable，一般是可训练的；如果不可训练，就是始终使用pretrained
            # embedding
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            # paragraph
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)
            # question
            # 把p和q中代表词的int转换为embedding

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        p和q都做了encode，之后又合并了，本质上不是encode，是预处理
        """
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.p_length,
                                        self.hidden_size)
            # self.p_emb是二维的，一个维度是batch size，另一个维度是多个sample中最长的
            # p的长度（三个维度？还有一个维度是word embedding的维度）；
            # self.p_length是一维的，长度是batch size，具体内容是各个sample的
            # p的长度；hidden_size是这个rnn中lstm的hidden unit数目，是可以调参的
            # rnn的返回值既有output，也有hidden state，此处只记录output
            # 其实是从一个矩阵变换到了另一个矩阵
            # print('The shpae of sep encodes:',
            #       self.sep_p_encodes.get_shape())
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn('bi-lstm', self.q_emb, self.q_length,
                                        self.hidden_size)
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes,
                                               self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes,
                                               self.dropout_keep_prob)

    def _match(self):
        """
        The core of RC model, get the question-aware passage
        encoding with either BIDAF or MLSTM
        The attention process
        文档的加权句向量，权重由问句决定
        """
        if self.algo == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        else:
            raise NotImplementedError(
                'The algorithm {} is not implemented.'.format(self.algo))
        self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes,
                                                    self.sep_q_encodes,
                                                    self.p_length,
                                                    self.q_length)
        # 所谓的attention的过程，就是把两个矩阵用神经网络的方式合并成一个矩阵
        # 中间过程要拿到attention分布，这个分布与q有关，然后把这个分布施加在p上，得到新的
        # 矩阵; p后来要被pointer net来用，所以p是主要的，q是次要的，p和q是有关系的，
        # 但这种关系如何建模，一般是采取attention的方式来建模，不管MLSM还是DIDAF，
        # 用的都是这种方式，只不过细节不同
        # 只记录lstm的outputs，不记录hidden states
        # print('The shpae of match encodes:',
        #       self.match_p_encodes.get_shape())
        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes,
                                                 self.dropout_keep_prob)

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        得到新的rnn
        """
        with tf.variable_scope('fusion'):
            self.fuse_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes,
                                         self.p_length,
                                         self.hidden_size, layer_num=1)
            # attention之后，再用bi-lstm做一次矩阵变换，是真正的encode
            # 此处lstm的layer_num是可调的（layer_num大于1的话，似乎代码有bug）
            # 同样只记录outputs，作为pointer net选择的对象
            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes,
                                                    self.dropout_keep_prob)
            # print('hidden_size in _fuse in rc_model.py:', self.hidden_size)
            # print('The shpae of fuse encodes:',
            #       self.fuse_p_encodes.get_shape())

    def _decode(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same
        document.
        And since the encodes of queries in the same document is same,
        we select the first one.
        """
        with tf.variable_scope('same_question_concat'):
            # print('The shape of fuse encodes:',
            #       self.fuse_p_encodes.get_shape())
            batch_size = tf.shape(self.start_label)[0]
            concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes,
                [batch_size, -1, 2 * self.hidden_size]
            )
            # 维度调整，沿hidden_size方向延伸，由于前面在产生passage_encodes使用的就
            # 是concat模式，所以这一步其实没啥效果，本来最后一维就是300维
            # 如果产生passage_encodes使用的是average模式，那就更不该有这一步，若是有
            # 这一步，就出错了
            # 文章信息，其实是把q合并在p上之后的信息

            # print('The shape of q1 encodes:',
            #       self.sep_q_encodes.get_shape())
            no_dup_question_encodes = tf.reshape(
                self.sep_q_encodes,
                [batch_size, -1, tf.shape(self.sep_q_encodes)[1],
                 2 * self.hidden_size]
                # 三维变四维，现在的第三维变成了原来的第二维
                # tf.shape(self.sep_q_encodes)[1]，现在的第二维事实上只有一行
            )[0:, 0, 0:, 0:]
            # 问题信息，四个维度，其中第二个维度只取了一列数据，结果还是三维的
            # 同样，这一步也没啥效果
            # print('The shape of q2 encodes:',
            #       self.sep_q_encodes.get_shape())
        decoder = PointerNetDecoder(self.hidden_size)
        # self.fw_outputs, self.fw_outputs2, self.bw_outputs = \
        #     decoder.decode2(concat_passage_encodes, no_dup_question_encodes)
        # self.fw_cell, self.bw_cell, self.fw_cell1 = \
        #     decoder.decode2(concat_passage_encodes, no_dup_question_encodes)
        self.start_probs, self.end_probs = decoder.decode(
            concat_passage_encodes, no_dup_question_encodes)
        # decoder也是用的bi-lstm，no_dup_question_encodes作为hidden states输入，
        # input是一个fake input，concat_passage_encodes是pointer net所用的矩阵，
        # 是分类的时候用的，开始位置有一次分类，结束位置有一次分类

        # 最终算出来的是passage中各个位置充当start的概率，和充当end的概率
        # 具体怎么对待这两个概率，可以有多种规则，多种算法
        # forword propagation到这里就完成了

    def _compute_loss(self):
        """
        The loss function
        计算损失函数就为了之后的参数优化，本质上是为了back propagation
        此处损失函数用的还是cross entropy，既考虑start loss，也考虑end loss，把
        二者结合起来。在做拟合和推测时，都用到start prob和end prob，但处理方法是不同的
        还有一种处理方法，更类似加强学习，就是拟合的结果不和answer相比，而是把目标
        函数变成start_prob*end_prob的最大化
        """

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                # 此处是name_scope
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                # labels是具体位置的序号，此处要one hot encoding
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
                # cross entropy的公式, + epsilon是为了处理probs为0的情况
            return losses

        self.start_loss = sparse_nll_loss(probs=self.start_probs,
                                          labels=self.start_label)
        self.end_loss = sparse_nll_loss(probs=self.end_probs,
                                        labels=self.end_label)
        # 要算两个cross entropy的loss，start和end各算一次
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        # 优化的目标函数：两个loss的加和求最小值
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
                # 做l2 regularization，使得拟合的weight不至于太大
            self.loss += self.weight_decay * l2_loss
            # self.weight_decay就是做l2 regularization时前面的那个系数，
            # 用来控制正则化的程度

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(
                self.learning_rate)
        else:
            raise NotImplementedError(
                'Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)
        # 直接上minimize函数最省事

    def _train_epoch(self, train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0
        # 这一个epoch中总共训练了多少样本
        # log_every_n_batch, n_batch_loss = 50, 0
        log_every_n_batch, n_batch_loss = 30, 0

        for bitx, batch in enumerate(train_batches, 1):
            # print('bitx in rc_model.py', bitx)
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: dropout_keep_prob}
            # print('shape of self.start_probs:',
            #       self.start_probs.get_shape().as_list())
            # print(self.start_probs)
            # print('shape of self.end_probs:',
            #       self.end_probs.get_shape().as_list())
            # print(self.end_probs)
            # Both are tensors
            # start = self.sess.run(self.start_probs, feed_dict)
            # end = self.sess.run(self.end_probs, feed_dict)
            # print('shape of start:', start.shape)
            # print('start:', start)
            # print('shape of end:', end.shape)
            # print('end:', end)
            # print('start == end:', start == end)
            # fw = self.sess.run(self.fw_outputs, feed_dict)
            # fw2 = self.sess.run(self.fw_outputs2, feed_dict)
            # bw = self.sess.run(self.bw_outputs, feed_dict)
            # print('fw:', fw)
            # print('fw2:', fw2)
            # print('bw:', bw)

            # 实例也是动态生成的?
            # print('fw_cell==bw_cell:', self.fw_cell == self.bw_cell)
            # print('type of fw_cell', type(self.fw_cell))
            # print('type of fw_cell1', type(self.fw_cell1))
            # print('fw_cell==fw_cell1:', self.fw_cell == self.fw_cell1)

            # loss = self.sess.run(self.loss, feed_dict)
            # print('loss in _train_epoch in rc_model.py:', loss)
            # print('All parameters in rc_model.py', self.all_params)

            # gradients_none = [gradient for gradient in gradients
            #                   if gradient[0] is None]
            # print('gradients_none in _train_epoch in rc_model.py:',
            #       len(gradients_none))

            # results_g = self.sess.run(self.capped_gradients[1][0], feed_dict)
            # print('results_g in _train_epoch in rc_model.py:', results_g)

            # self.logger.debug(self.sess.run(tf.shape(self.p_emb), feed_dict))
            # self.logger.debug(
            #     self.sess.run(tf.shape(self.sep_p_encodes), feed_dict))
            # self.logger.debug(
            #     self.sess.run(tf.shape(self.match_p_encodes), feed_dict))
            # self.logger.debug(
            #     self.sess.run(tf.shape(self.fuse_p_encodes), feed_dict))

            # self.logger.debug(
            #     self.sess.run(tf.shape(concat_passage_encodes), feed_dict))
            # self.logger.debug(
            #     self.sess.run(tf.shape(no_dup_question_encodes), feed_dict))

            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
            # variable自动更新，返回的也是更新后的variable，这里就不记录了

            total_loss += loss * len(batch['raw_data'])
            # loss是根据batch size平均后的结果，这里进行加总
            total_num += len(batch['raw_data'])
            # 累加batch size，或者最后一批剩下的数目
            print('total_num in rc_model.py', total_num)
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info(
                    'Average loss from batch {} to {} is {}'.format(
                        bitx - log_every_n_batch + 1,
                        bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    # 打印的是n_batch的平均loss，返回的是整个epoch的平均loss

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each
              epoch
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        # padding token的id
        max_bleu_4 = 0
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id,
                                                  shuffle=True)
            train_loss = self._train_epoch(train_batches, dropout_keep_prob)
            self.logger.info(
                'Average train loss for epoch {} is {}'.format(epoch,
                                                               train_loss))

            if evaluate:
                self.logger.info(
                    'Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches(
                        'dev', batch_size, pad_id, shuffle=False)
                    eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Bleu-4'] > max_bleu_4:
                        self.save(save_dir, save_prefix)
                        # 保存模型
                        max_bleu_4 = bleu_rouge['Bleu-4']
                else:
                    self.logger.warning(
                        'No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None,
                 save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved
        if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers,
            answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to
            raw sample and saved
        """
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: 1.0}
            # evaluate必然是没有dropout的
            start_probs, end_probs, loss = self.sess.run(
                [self.start_probs, self.end_probs, self.loss], feed_dict)
            # self.logger.debug(start_probs)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            padded_p_len = len(batch['passage_token_ids'][0])
            # self.p中最长的那个样本的长度？batch['passage_token_ids']应该是个
            # 多维的np.ndarray才对，有可能是当list of list来处理，就是第一行,
            # 也就是第一个样本

            for sample, start_prob, end_prob in zip(batch['raw_data'],
                                                    start_probs, end_probs):

                best_answer = self.find_best_answer(sample, start_prob,
                                                    end_prob, padded_p_len)
                # 在做evaluate和test的推测工作时，要这样利用start_prob和end_prob
                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    pred_answers.append({'question_id': sample['question_id'],
                                         'question_type':
                                             sample['question_type'],
                                         'question': sample['question'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                        'question_type':
                                            sample['question_type'],
                                        'answers': sample['answers'],
                                        'entity_answers': [[]],
                                        'yesno_answers': []})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w', encoding='utf8') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer,
                                          ensure_ascii=False) + '\n')

            self.logger.info(
                'Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set,
        # since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num

        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    # 利用utils包，normalize strings to space joined chars
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
            # pred_dict是预测值，ref_dict是真实值
            # 利用utils包，calculate bleu and rouge metrics
        else:
            bleu_rouge = None
        return ave_loss, bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len):
        """
        Finds the best answer for a sample given start_prob and end_prob for
        each position.
        This will call find_best_answer_for_passage because there are multiple
        passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            # 每篇document选了一段
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            # 如果passage长度超过self.max_p_len，在这儿会进行处理
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if score > best_score:
                # 来自不同文章的答案可以比较，因为能算出score
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]:
                best_span[1] + 1])
            # best_span就是个2个元素的向量
        return best_answer

    def find_best_answer_for_passage(self, start_probs, end_probs,
                                     passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob
        from a single passage
        passage_len是start和end之间最长的长度？
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
            # 又是取最小处理
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                # end_idx必须在start_idx后面
                if end_idx >= passage_len:
                    continue
                    # start_idx与end_idx的间隔不能太长
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob
    # 果然是只有两个元素的tuple

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info(
            'Model saved in {}, with prefix {}.'.format(model_dir,
                                                        model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model
        indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info(
            'Model restored from {}, with prefix {}'.format(model_dir,
                                                            model_prefix))
