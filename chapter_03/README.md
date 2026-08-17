# Chapter 3: Coding Attention Mechanism

This chapter works through implementations of self-attention of increasing
complexity:

- Simplified Self-Attention
- Self-Attention
- Causal Attention
- Multi-Head Attention

## An Aside to RNNs

When we talk about architectures that have an _Encoder_ and _Decorder_, the job
of the encoder is to read in and process the entire text. Whereas the job of the
decoder is to produces a corresponding response as text (e.g. a translation).

RNNs (Recurrent Neural Networks) are a popular encoder-decoder architecture.
This is one approach that was being used before the Transformer architecture
existed. For instance, in 2015 (two years before [_Attention Is All You
Need_](https://arxiv.org/abs/1706.03762)) Andrej Karpathy published a blog post
called [The Unreasonable Effectiveness of Recurrent Neural
Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/). GPT-like
LLMs use a decoder-only transformer.

From this [Recurrent Neural Network
cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks/):

> Recurrent neural networks, also known as RNNs, are a class of neural networks
> that allow previous outputs to be used as inputs while having hidden states.

As the book puts it:

> The encoder updates its hidden state (the internal values at the hidden
> layers) at each step, trying to capture the entire meaning of the input
> sentence in the final hidden state.

The decoder goes through a similar process using this final hidden state as its input:

> The decoder then takes this final hidden state to start generating the
> translated sentence, one word at a time. It also updates its hidden state at
> each step, which is supposed to carry the context necessary for the next-word
> prediction.

The problem with this architecture is that context and meaning can get lost when
the decoder only has this final hidden state to go off of.

## Self Attention

An attention mechanism for RNNs was [introduced in 2014 by Bahdanau et
al](https://arxiv.org/abs/1409.0473) which allows the decoder to access input
tokens. The input tokens are annotated with _attention weights_ which suggest
each of their relative importance.

> Self-attention is a mechanism that allows each position in the input sequence
> to consider the relevancy of, or "attend to," all other positions in the same
> sequence when computing the representation of a sequence.

The _self_ in self-attention has to do with each input token being assigned an
attention weight relative to other tokens in the self-same input sequence. This
is different than an attention model that attention weights are computed across
sequences (like input and output).

### Simplified Self-Attention

A version of self-attention that eschews trainable weights to make some key
concepts easier to understand.

To power self-attention, we need to compute a _context vector_ for every single
element (token / embedding) of the input sequence. A context vector for, say,
the second element in the input sequence will be computed with respect to the
first, third, and all other elements in the input sequence. The context vector
is an embedding itself -- that is, a d-dimensional vector like the token
embeddings.

The idea here is that these _context vectors_ provide contextual signals
(relationship and relevance) about how each part of the input relates to other
parts of the input.

To compute _context vectors_ for each input token, we need to go through a
series of steps.

First, we need to compute the _attention scores_ for each input token. That is
done, for a given input token, by computing a new vector where each position is
the dot product of the input token at that index with the given input token. The
resulting vector, the attention score, will have the same dimensionality as the
input sequence size.

> Compute the attention scores as dot products between the inputs.

We then translate the attention scores into _attention weights_. This is a
process where we use the _softmax_ function to convert each of those vectors
into normalized weights where the sum of all values in the vector equal _1_.

> The attention weights are a normalized version of the attention scores.

The normalized attention weights are then used to calculate the _context vector_
for each input token. This is done by multiplying the attention weight positions
with each corresponding input token embedding, and then summing all of those up.
The result of each of these is a vector with the same length (shape) as the
embeddings.

> The context vectors are computed as a weighted sum over the inputs.

### Self-Attention

The self-attention mechanism that is used in GPT models is specifically referred
to as _scaled dot-product attention_.

This method introduces trainable weights which help the model produce "good"
context vectors.
