# Build an LLM

Code examples and implementations as I read and work through [Build a Large
Language Model from scratch](http://mng.bz/orYv). The book's supplementary repo
can be found [here](https://github.com/rasbt/LLMs-from-scratch).

## BPE (Byte Pair Encoding) Tokenizer

An example of running this with various CLI parameters:

```bash
❯ python3 bpe_tokenizer.py \
    --corpus ~/dev/jbranchaud/til/combined.md \
    --output ./output.txt \
    --special-tokens '<|endoftext|>'
```
