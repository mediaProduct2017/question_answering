# multi-pointer-generator

Answer embedding and answer generation: pointer-generator的核心是attention，训练的时候输入是真正的answer，输出是预测的answer；预测的时候，输入从start的词向量开始，不断加入预测出来的词的词向量，不断输出新预测的词

训练阶段：

```flow
input=>inputoutput: answer: static word embedding sequence (n*demb matrix)
nn=>operation: 单层神经网络
answer1=>inputoutput: answer: static word embedding sequence (n*d matrix)
wordvec=>subroutine: produce word vectors as Q
self=>operation: self-attention: self as K and V
answer2=>inputoutput: answer with long-distance interactions
pe=>operation: PE[t,k] (n*d matrix)
answer3=>inputoutput: Appr: answer with long-distance interaction and position info (n*d matrix)
wordvec2=>subroutine: produce word vectors as Q multiple times by different methods or parameters
self2=>operation: self-attention: self as K and V (multiHead)
add=>subroutine: add the different matrices
answer4=>inputoutput: Amha: answer with long-distance interaction and position info after  multihead self attention (n*d matrix)
wordvec3=>subroutine: produce word vectors as Q multiple times by different methods or parameters
att=>operation: attention by context matrix: context as K and V (multiHead)
add2=>subroutine: add the different matrices
answer5=>inputoutput: Aac: answer with long-distance interaction, position info and context info after  multihead attention (n*d matrix)
add2=>subroutine: 前馈网络，做融和：Add the matrices above
answer6=>inputoutput: Amerge: Appr + Amha + Aac (n*d matrix)
feed=>operation: feed forward network (普通神经网络)
answer7=>inputoutput: Afeed
add3=>subroutine: 前馈网络，做融和：Add the matrices; Residual feedforward network
answer8=>inputoutput: Aself: Afeed + Amerge (n*d matrix)
aselft=>operation: Aself(t-1): length d vector
lstm1=>operation: LSTM: 包含反馈输入
ht=>inputoutput: h(t): length d vector
transform=>subroutine: 做线性变换
wht=>inputoutput: W2*h(t): length d vector
wordvec4=>subroutine: produce word vector as Q
att2=>operation: attention by context matrix: context as K (we don't need V here)
weights=>inputoutput: attention weights after softmax: length l vector
att3=>subroutine: attention final step: weighted sum of V (context)
vec5=>inputoutput: attention result C(T)alpha(T): length d vector
transform2=>subroutine: 做连接，然后做线性变换
vec6=>inputoutput: W(C(T)alpha(T); h(t))
tanh=>subroutine: tanh
vec7=>inputoutput: c(t), c(t-1): length d vector

input->nn->answer1->wordvec->self->answer2->pe->answer3->wordvec2->self2->add->answer4
answer4->wordvec3->att->answer5->add2->answer6->feed->answer7->add3->answer8->aselft
aselft->lstm1->ht->transform->wht->wordvec4->att2->weights->att3->vec5->transform2->vec6
vec6->tanh->vec7->lstm1
```

答案产生: context and question

```flow
weights=>inputoutput: attention weights at time t after softmax: length l vector
merge=>subroutine: if context (question) has replicated words, add the attention weights of the same word
vec1=>inputoutput: attention probablities of word sequence: length l(prime) vector

weights->merge->vec1

```

答案产生：其他词汇，比如之前回答出现的词汇，共有n个unique words

 ```flow
vec7=>inputoutput: c(t): length d vector
transform=>subroutine: 线性变换，向量维度变化
vec8=>inputoutput: length v vector
soft=>subroutine: softmax
vec9=>inputoutput: probabilities of word sequence: length v vector

vec7->transform->vec8->soft->vec9
 ```

计算context, question, other的权重系数

```flow
transform=>subroutine: 做融合：连接
vec1=>inputoutput: c(t);h(t);Aself(t-1)
transform2=>subroutine: 线性变换
vec2=>inputoutput: Wpv*(c(t);h(t);Aself(t-1))
sigmoid=>subroutine: sigmoid
gamma=>inputoutput: the weight of other words

transform->vec1->transform2->vec2->sigmoid->gamma
```

