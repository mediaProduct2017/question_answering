# question_answering


(1) 机器阅读理解多种算法的比较，评测的数据使用stanford squad 1.1数据集（英文数据集），评测的运行时间大概是1小时左右，使用的gpu是一个GeForce GTX 1080

发现decanlp的效果要好的多

squad 1.1 training dataset: 87599 questions    
squad 1.1 validation dataset: 10570 questions

英文词汇数：102423，其中，只出现一次的词4472，出现两次或两次以上的词97951

algorithm: decanlp  
batch size: 128 (查代码后发现，train所用的batch size与validation相同); validation batch size: 128  
Without pretrained word embeddings (When using glove as pretrained word embeddings, performance is better)  
python decaNLP/train.py --train_tasks squad --gpus 0 --train_batch_tokens 5000 --val_batch_size 128  
There are 14,469,902 parameters in the model    

Result:  
time: 1小时（1个gpu，GeForce GTX 1080）  
'Bleu-1': 0.525  
'Bleu-4': 0.329  
'Rouge-L': 0.493  

time: 3.5小时（1个gpu，GeForce GTX 1080）  
'Bleu-1': 0.643  
'Bleu-4': 0.500  
'Rouge-L': 0.636  

algorithm: BIDAF 完整模型

[squad_mlstm](https://github.com/arfu2016/DuReader/tree/master/squad_mlstm)

/Users/arfu/Documents/Python_projects/PycharmProjects/DuReader

epochs: 2  
batch size: 128  
No pretrained word embeddings  
python squad_mlstm/run_python3.py --train --algo BIDAF --epochs 2 --batch_size 128 --gpu "0"  
There are 32,879,103 parameters in the model  

Result:  
time: 1小时(1个gpu，GeForce GTX 1080)  
'Bleu-1': 0.132  
'Bleu-4': 0.083  
'Rouge-L': 0.349 

algorithm: MLSTM 完整模型

[squad_mlstm](https://github.com/arfu2016/DuReader/tree/master/squad_mlstm)

epochs: 2    
batch size: 128  
No pretrained word embeddings  
python squad_mlstm/run_python3.py --train --algo MLSTM --epochs 2 --batch_size 128 --gpu "0" 
There are 33,647,105 parameters in the model

Result:  
time: 1.2小时(1个gpu，GeForce GTX 1080)  
'Bleu-1': 0.149  
'Bleu-4': 0.0885  
'Rouge-L': 0.345  

algorithm: BIDAF 简化模型

[squad2](https://github.com/arfu2016/DuReader/tree/master/squad2)

epochs: 4    
batch size: 32    
No pretrained word embeddings  
python squad2/run_python3.py --train --algo BIDAF --epochs 4 --batch_size 128 --gpu "0"   
There are 30,176,403 parameters in the model

Result:
time: 1小时(1个gpu，GeForce GTX 1080)  
'Bleu-1': 0.086  
'Bleu-4': 0.047  
'Rouge-L': 0.172

algorithm: MLSTM 简化模型

[squad2](https://github.com/arfu2016/DuReader/tree/master/squad2)

epochs: 2  
batch size: 128  
No pretrained word embeddings  
python squad2/run_python3.py --train --algo MLSTM --epochs 2 --batch_size 128 --gpu "0"  
There are 32,023,505 parameters in the model

Result:  
time: 1小时(1个gpu，GeForce GTX 1080)  
'Bleu-1': 0.0887  
'Bleu-4': 0.0554  
'Rouge-L': 0.2736  


(2) 机器问答和机器阅读理解目前可用模型的举例：

1. Seqence-to-sequence model，可用于一问一答的闲聊模块

[seq2seq_model](https://github.com/arfu2016/seq2seq_model)

[llSourcell/tensorflow_chatbot](https://github.com/arfu2016/tensorflow_chatbot)

[llSourcell/chatbot_tutorial](https://github.com/llSourcell/chatbot_tutorial)

[Conchylicultor/DeepQA](https://github.com/arfu2016/DeepQA)

2. Dynamic memory networks，机器阅读理解

[llSourcell/How_to_make_a_chatbot](https://github.com/llSourcell/How_to_make_a_chatbot)

[Implementing Dynamic memory networks](https://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/)

[cs224n: dynamic neural networks for question answering](https://www.youtube.com/watch?v=T3octNTE7Is)

3. Pointer network，机器阅读理解的decoder

[DuReader](https://github.com/arfu2016?tab=repositories)

4. MQAN (Multitask Question Answering Network)，多任务模型，可用于机器阅读理解

[salesforce/decaNLP](https://github.com/arfu2016/decaNLP)


（3）机器阅读理解的方法尝试用于中文信息抽取

Subset of Baidu Machine Reading Dataset (中文):  
trianing: 82909 questions  
validation: 5000 questions  

algorithm: BIDAF 完整模型  
epochs: 1  
batch size: 32  
No pretrained word embeddings  
python run.py --train --algo BIDAF --epochs 1 --batch_size 32 \  
--train_files '../data/preprocessed/trainset/search.train2.json' \  
--dev_files '../data/preprocessed/devset/search.dev2.json'  

There are 75,611,403 parameters in the model

一个epoch

Result:  
time: 9小时(32线程服务器的cpu，大概用掉20个线程；8G的gpu显存是不够的)  
'Bleu-1': 0.361  
'Bleu-4': 0.2236  
'Rouge-L': 0.2923  

两个epoch

Result:  
time: 18小时(32线程服务器的cpu，大概用掉20个线程；8G的gpu显存是不够的)  
'Bleu-1': 0.381  
'Bleu-4': 0.2393  
'Rouge-L': 0.3025  

python question_answering/ask_and_answer_messi.py --predict --algo BIDAF --batch_size 32    
给出一组context，从中选出最适合回答某个问题的context中的一段内容    
给出一组context，在其中掺入另外的一句回答，给定某个问题，选出这一组context当中比新掺入    
的这句更合适的回答，回答是context当中的一部分内容  

目前，使用这种方法的效果很一般，还需要进一步的研究

机器阅读理解技术用于搜索：  
以往的搜索都是建立在相似逻辑上的，现在的自动问答体现了相似之外的其他的推理逻辑

把自动问答用在搜索上有两个方案：

方案一：

首先用传统词匹配的方法定位与问题相似的文章，或者用分类器的办法把问题进行分类，每个类别是一篇或者几篇文章。

然后用baidu reader或者decanlp的算法，给定问题，从文章的每一段中都摘出一句或者几句话，摘出的
话都带着自己的概率，文章有好几段话，对摘出的概率进行比较，概率最大的就是最终的结果。

方案二：

首先用传统词匹配的方法定位与问题相似的文章，或者用分类器的办法把问题进行分类，每个类别是一篇或者
几篇文章。

然后用google talk to books的算法，首先把问题转换成句向量，然后把文章转换成一棵词向量、句向量、
段向量组成的树，通过问句向量首先找词向量，看有没有合适的可以作为答案的，如果没有，再找句，再找
段，最后找文章。


（4）机器阅读理解的方法尝试用于英文信息抽取

未来的研究方向

