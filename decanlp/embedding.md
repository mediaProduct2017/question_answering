# 句向量

Context and Question embeddings

```flow
st=>start: 开始
e=>end: 结束

input=>inputoutput: sentence1: static word embedding sequence (n*demb matrix)
bilstm1=>operation: bilstm
syntactic=>inputoutput: sentence1: word embedding sequence with syntactic info
bilstm2=>operation: bilstm
semantic=>inputoutput: sentence1: word embedding sequence with sementic info
enter?=>condition: enter neural net to transform
wordvec=>inputoutput: word embedding as Q
lstm=>operation: lstm
othermodel=>operation: other model
ho?=>condition: lstm or other neural net
hidden=>inputoutput: hidden state as Q
outputvec=>inputoutput: output vector as Q
kv=>operation: sentence2 as K and V
attention=>inputoutput: attention vector
soft=>subroutine: normalized by softmax
attentionp=>inputoutput: attention probability vector
mul=>subroutine: multiply as weights of V
sen=>inputoutput: sentence1 with info from sentence2
bilstm3=>operation: bilstm
syntactic2=>inputoutput: sentence1 with sentence2: word embedding sequence with syntactic info
bilstm4=>operation: bilstm
semantic2=>inputoutput: sentence1 with sentence2: word embedding sequence with sementic info
wordvec2=>subroutine: produce word vectors as Q
self=>operation: self-attention: self as K and V
distance=>inputoutput: snetence1 merged by sentence2 with long-distance interactions
bilstm5=>operation: 双层bilstm
semantic3=>inputoutput: snetence1 merged by sentence2 with long-distance interactions and semantic info

input->bilstm1->syntactic->bilstm2->semantic->enter?
enter?(yes)->ho?
enter?(no)->wordvec->kv
ho?(yes)->lstm->hidden->kv
ho?(no)->othermodel->outputvec->kv
kv->attention->soft->attentionp->mul->sen->bilstm3->syntactic2->bilstm4->semantic2->wordvec2->self->distance
distance->bilstm5->semantic3
```

## 前半部分

```flow
input=>inputoutput: sentence1: static word embedding sequence (n*demb matrix)
bilstm1=>operation: bilstm
syntactic=>inputoutput: sentence1: word embedding sequence with syntactic info
bilstm2=>operation: bilstm
semantic=>inputoutput: sentence1: word embedding sequence with sementic info
enter?=>condition: enter neural net to transform
wordvec=>inputoutput: Q
lstm=>operation: lstm
othermodel=>operation: other model
ho?=>condition: lstm or other neural net
hidden=>inputoutput: hidden state as Q
outputvec=>inputoutput: output vector as Q

input->bilstm1->syntactic->bilstm2->semantic->enter?
enter?(yes)->ho?
enter?(no)->wordvec->kv
ho?(yes)->lstm->hidden->kv
ho?(no)->othermodel->outputvec->kv
```

## 后半部分

```flow
hidden=>inputoutput: hidden state as Q
kv=>operation: sentence2 as K and V
attention=>inputoutput: attention vector
soft=>subroutine: normalized by softmax
attentionp=>inputoutput: attention probability vector
mul=>subroutine: multiply as weights of V
sen=>inputoutput: sentence1 with info from sentence2
bilstm3=>operation: bilstm
syntactic2=>inputoutput: sentence1 with sentence2: word embedding sequence with syntactic info
bilstm4=>operation: bilstm
semantic2=>inputoutput: sentence1 with sentence2: word embedding sequence with sementic info
wordvec2=>subroutine: produce word vectors as Q
self=>operation: self-attention: self as K and V
distance=>inputoutput: snetence1 merged by sentence2 with long-distance interactions
bilstm5=>operation: 双层bilstm
semantic3=>inputoutput: snetence1 merged by sentence2 with long-distance interactions and semantic info

hidden->kv
kv->attention->soft->attentionp->mul->sen->bilstm3->syntactic2->bilstm4->semantic2->wordvec2->self->distance
distance->bilstm5->semantic3

```

