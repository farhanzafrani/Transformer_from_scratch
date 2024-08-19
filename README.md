# Transformer_from_scratch

Before going through code, I think that, it's important to specify the different dimensions that are used in the  `Attention all you need` paper.
 - d_model --> dimension of the embedding vector
 - vocab_size --> number of words in vocabulary
 - seq_Len --> dimension of the input sequence or the sentence
 


 ## What is Embedding?

In the **"Attention is All You Need"** paper (which introduced the **Transformer model**), **embedding** refers to the process of mapping input tokens (like words or characters) into dense, continuous vector representations. These embeddings capture semantic information about the tokens and allow the model to process them in a more meaningful way.

## Key Concept: Why Embeddings are Necessary

Words or tokens in natural language processing (NLP) are discrete, but models work with continuous numerical data. To bridge this gap:
- Each token (word, subword, or character) is transformed into a **fixed-size vector**.
- These vectors are learned during training and capture relationships between tokens (e.g., synonyms have similar embeddings).

For example, the word "king" might be represented by a vector like:
`[0.27, 0.64, -0.52, 0.12, ...]` # a 512-dimensional vector


## Embedding in the Transformer Model

In the Transformer architecture, embeddings are used at the **input** and **output** stages:
1. **Input Embedding**: The input tokens are embedded into vectors of a fixed dimension (`d_model`).
2. **Positional Encoding**: Since the Transformer has no inherent sense of sequence, **positional encodings** are added to the embeddings to provide information about the order of tokens.

Let’s break down the embedding process mathematically.

## Embedding Formula

Let:
- \( V \) be the size of the vocabulary (total number of unique tokens).
- \( d_{\text{model}} \) be the dimension of the embedding space.

For each token \( x_i \), the embedding layer learns a matrix \( E \in \mathbb{R}^{V \times d_{\text{model}}} \) such that the embedding for token \( x_i \) is:

\[
\mathbf{e}(x_i) = E[x_i]
\]

Where:
- \( E[x_i] \) is the \( i \)-th row of the embedding matrix, which corresponds to the token’s vector representation.

## Positional Encoding

To incorporate the order of tokens, **positional encodings** are added to the embeddings. This is necessary because the attention mechanism doesn’t inherently account for the order of words in a sequence.

The **positional encoding** for a position \( pos \) and dimension \( i \) is defined as:

```math
\[
\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]
\[
\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]
```

Where:
- \( pos \) is the position of the token in the sequence.
- \( i \) refers to the dimension within the embedding vector.

These positional encodings are added to the original token embeddings:

```math
\[
\text{Embedding}(x_i) = E[x_i] + \text{PE}(pos)
\]
```

This gives the model information about both the content of the token and its position in the sequence.

## Embeddings in Action

Once embeddings are created and adjusted by positional encodings:
- They are fed into the Transformer’s **self-attention mechanism**.
- The model learns to assign **attention weights** to different parts of the input sequence based on the relationships captured in the embeddings.

This embedding process is crucial to enabling the model to understand both the content of each token and the structural relationships within the sequence.

