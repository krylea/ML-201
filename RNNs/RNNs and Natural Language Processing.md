#RNNs and Natural Language Processing

##RNN Cells
One of the most important architecture choices when it comes to RNNs is what to use as the base RNN cell. The most common options are a) traditional Elman RNN cells, b) Long Short-Term Memory cells (or LSTMs), or c) Gated Recurrent Units (or GRUs). Here are some heuristics you can use as to when each of these might be appropriate to use:
a) Elman RNN cells. Don't use these. Ever. Really, really don't.
b) LSTMs. 9 times out of 10, these are going to be your best bet. LSTMs blow other models out of the water when it comes to learning long-term causal relationships in sequential data. Their only disadvantages are:
  i) Complexity - if you are new to the field, LSTMs may seem somewhat complicated. They have two hidden states rather than one to deal     with, their inner "gate" operations can seem daunting when you first look at them, and some of the simplified things you learn about Elman cells may not necessarily apply.
  ii) Training time - LSTMs can take a bit longer to train than other models due to their increased number of computations.
In spite of these downsides, LSTMs are still great, and you should use them. Make them bidirectional to get even more out of them.
c) GRUs. These are kind of a weird one. They came into the scene much later than the other models, and it is often hard to evaluate the difference in performance between them and LSTMs. The intuition I have gathered indicates that in a lot of tasks, LSTMs are just better - especially if you are dealing with long sequences. It is, however, very difficult to say for sure one way or the other.

One last note on this topic - when you are first presented with RNNs, almost all the high-level descriptions you will get will follow the following rough outline:
1) The model takes in an input vector x and a previous hidden state h
2) The model computes a new hidden state using weight matrices acting on these inputs, combined with an activation function
3) The model produces an output through another weight matrix acting on the current hidden state, combined with a second activation. This output, along with the new hidden state, are the outputs of the layer.
This is really only correct for Elman cells. In GRUs, the hidden state is itself simply taken to be the output. In LSTMs, the first hidden state h (not the cell state c) is taken to be the output. If you want, you can stack further layers on this yourself to compute an output vector, but most existing frameworks will *not* include that automatically.

##Batching, Padding and Masking
This is perhaps the most important thing that I was never taught about RNNs (and to some extent deep learning in general). Almost any resource you look at that describes a model - whether it is a blog post, a paper, or something else - will describe the model's architecture in terms of how it acts on a single input. This is perfectly fine when it comes to performing inference, or training using "pure" stoachastic gradient descent. But, in practice, you *never* actually want to use pure SGD. You *always* want to make use of GPU parallelization with minibatch stochastic gradient descent. And implementing a model with batched training *properly* is *much, much, much* harder than implementing it for inference, especially when it comes to RNNs.

##Encoder-Decoder Models (e.g. Sequence to Sequence)
