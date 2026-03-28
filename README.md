# Build an LLM

Code examples and implementations as I read and work through [Build a Large
Language Model from scratch](http://mng.bz/orYv). The book's supplementary repo
can be found [here](https://github.com/rasbt/LLMs-from-scratch).

## BPE (Byte Pair Encoding) Tokenizer

Throughout the book we will be using
[`tiktoken`](https://github.com/openai/tiktoken) as our BPE tokenizer. I was
curious to learn more about this tokenization process and how I might (naively)
implement my own. Even though I'll be moving forward with the much more robust
set of tokens from `tiktoken`, I enjoyed building my own.

An example of running this with various CLI parameters:

```bash
❯ python3 bpe_tokenizer.py \
    --corpus ~/dev/jbranchaud/til/combined.md \
    --output ./output.txt \
    --special-tokens '<|endoftext|>'
```
