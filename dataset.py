"""
@Project   : DuReader
@Module    : dataset.py
@Created   : 7/23/18 5:50 PM
@Desc      :
"""

import json
from collections import Counter

import jieba
import numpy as np
from work4.logger_setup import define_logger
logger = define_logger('work3.dataset')


class BRCDataset:
    """
    This module implements the APIs for loading and using baidu reading
    comprehension dataset
    """

    def __init__(self, max_p_num, max_p_len, max_q_len,
                 train_files=[], dev_files=[], test_files=[]):
        # p: paragraph? q: question?
        self.logger = logger
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file, train=True)
                # 准备训练数据：train_set中放的是读入的数据
            self.logger.info(
                'Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file)
            self.logger.info(
                'Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info(
                'Test set size: {} questions.'.format(len(self.test_set)))

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path, encoding='utf-8') as fin:
            data_set = []
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                # 把json格式转换成python格式
                if train:
                    if len(sample['answer_spans']) == 0:
                        continue
                    if sample['answer_spans'][0][1] >= self.max_p_len:
                        continue
                    # 对容错的考虑
                    # 如果没有'answer_spans'，该行数据就被舍弃了，
                    # 不放在training data中

                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']
                    # 重新命名

                sample['question_tokens'] = sample['segmented_question']
                # 重新命名

                sample['passages'] = []
                for d_idx, doc in enumerate(sample['documents']):
                    # 每一篇document都做这样的处理
                    if train:
                        most_related_para = doc['most_related_para']
                        # 文章中最相关的段落
                        # The most related paragraphs are selected according to
                        # highest recall of the answer tokens of each document,
                        # and the index of the selected paragraph of each
                        # document is stored in "most_related_para".
                        # 如果没有'most_related_para'这一项如何处理，
                        # 也应该有所设计： 从most_related_para的获取过程看，
                        # 有answer必有'most_related_para'
                        # 若没有answer, 就没有'answer_spans'，
                        # 在前面已经处理过了，continue
                        sample['passages'].append(
                            {'passage_tokens':
                                 doc['segmented_paragraphs'][most_related_para],
                             # 只保留最相关段落的token
                             'is_selected': doc['is_selected']}
                            # 记录该篇文章是否被问题回答者参考
                        )
                        # 只保留了最相关的段落
                    else:
                        para_infos = []
                        for para_tokens in doc['segmented_paragraphs']:
                            question_tokens = sample['segmented_question']
                            common_with_question = Counter(
                                para_tokens) & Counter(question_tokens)
                            correct_preds = sum(common_with_question.values())
                            if correct_preds == 0:
                                recall_wrt_question = 0
                            else:
                                recall_wrt_question = float(
                                    correct_preds) / len(question_tokens)
                            para_infos.append(
                                (para_tokens, recall_wrt_question,
                                 len(para_tokens)))
                        para_infos.sort(key=lambda x: (-x[1], x[2]))
                        fake_passage_tokens = []
                        for para_info in para_infos[:1]:
                            fake_passage_tokens += para_info[0]
                        sample['passages'].append({'passage_tokens':
                                                   fake_passage_tokens})
                data_set.append(sample)
                # 把该行数据加入到data_set中
        return data_set

    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        # logger.debug(indices)
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}
        # 这里用来对语料做标记，主要是标记start_id和end_id

        # for sample in batch_data['raw_data']:
        #     logger.debug(sample['question'])
        max_passage_num = max(
            [len(sample['passages']) for sample in batch_data['raw_data']])
        # 这一批数据中段落数目最大的样本
        max_passage_num = min(
            self.max_p_num, max_passage_num)
        # 设定的最大的段落数
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                # pidx表示sample中的第几个段落，比如第0个、第1个等
                if pidx < len(sample['passages']):
                    batch_data['question_token_ids'].append(
                        sample['question_token_ids'])
                    batch_data['question_length'].append(
                        len(sample['question_token_ids']))
                    passage_token_ids = sample[
                        'passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(
                        passage_token_ids)
                    # 一般来说，max_passage_num是5，也就是每个问题对应着5段文档
                    # 所以batch_data['passage_token_ids']的维度是(16*5, 500)
                    batch_data['passage_length'].append(
                        min(len(passage_token_ids), self.max_p_len))
                    # 'passage_length'最大按self.max_p_len算
                    # 之后把这个参数传给tensorflow以后，长度之外的，tf就不管了
                else:
                    # 没有这么多的段落的话，补空值
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(
            batch_data, pad_id)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                # answer docs给出来以后，还要知道是哪个段落
                # 给出了answer来源于的passage，可用来设定start_id和end_id
                # 对于train和validate数据，'answer_passages'这个字段是必需的
                gold_passage_offset = padded_p_len * sample[
                    'answer_passages'][0]
                # 如果是16个batch，由于每个问题给出了5个document，开始的时候数据的第一维
                # 是16*5=80，但后来会缩减为16，padded_p_len是用于训练的每个文档的长度，
                # sample['answer_passages'][0]中放的是答案在第几个文档中的信息，所以
                # 二者相乘就是答案所在文档的起始点。

                # 这里，只取了sample['answer_passages']中的第一个元素，也就是说我们假
                # 设只有一个答案，实际上，dureader数据集是给出了多个答案的.
                # 如果考虑多个答案的话，最终的评分有可能会更高。
                batch_data['start_id'].append(
                    gold_passage_offset + sample['answer_spans'][0][0])
                # answer span给出的list以0开始，以answer长度终止
                batch_data['end_id'].append(
                    gold_passage_offset + sample['answer_spans'][0][1])
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        # pad后的长度考虑了self.max_p_len和self.max_q_len
        batch_data['passage_token_ids'] = [
            (ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
            for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [
            (ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
            for ids in batch_data['question_token_ids']]
        # 上面是padding batch_data的过程
        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Actually a generator with yield statement
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError(
                'No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                # 每个sample代表原数据文件中的一行
                for token in sample['question_tokens']:
                    yield token
                for passage in sample['passages']:
                    # sample中的每一个文档都做处理
                    for token in passage['passage_tokens']:
                        # tokens in the most related paragraph
                        yield token

    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
                # for training data, dev_set and test_set is None
            for sample in data_set:
                sample['question_token_ids'] = \
                    vocab.convert_to_ids(sample['question_tokens'])
                # convert tokens to ids
                for passage in sample['passages']:
                    passage['passage_token_ids'] = \
                        vocab.convert_to_ids(passage['passage_tokens'])

    def load_context(self, context, questions, vocab):
        context_set = []

        for paragraph, question in zip(context, questions):

            segmented_paragraphs = list(jieba.cut(paragraph))
            # 分词结果中可以包含标点符号

            sample = dict()

            sample['question_tokens'] = list(jieba.cut(question))
            # logger.debug(qa['question'])

            # question = ' '.join(sample['question_tokens'])
            sample_id = len(context_set)

            sample['passages'] = [
                {'passage_tokens': segmented_paragraphs,
                 'is_selected': True},
                {'passage_tokens': ['梅西', '技术', '好'],
                 # ['梅西', '迷失', '了']
                 # 梅西 啥 都 没 做
                 'is_selected': True}
            ]
            # logger.debug(sample['passages'][0]['passage_tokens'])

            sample['question'] = question
            # logger.debug(sample['question'])
            sample['question_id'] = sample_id
            sample['question_type'] = "DESCRIPTION"

            sample['match_scores'] = [1.00]

            sample['question_token_ids'] = \
                vocab.convert_to_ids(sample['question_tokens'])
            # convert tokens to ids
            for passage in sample['passages']:
                passage['passage_token_ids'] = \
                    vocab.convert_to_ids(passage['passage_tokens'])

            context_set.append(sample)

        return context_set

    def gen_mini_batches(self, set_name, batch_size, pad_id, context,
                         questions, vocab, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            context:
            questions:
            vocab:
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        context_set = self.load_context(context, questions, vocab)
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        elif set_name == 'ask':
            data = context_set
        else:
            raise NotImplementedError(
                'No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id)
