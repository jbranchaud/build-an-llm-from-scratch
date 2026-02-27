from collections import Counter
from pathlib import Path
from typing import TypeAlias

MergeRule: TypeAlias = tuple[tuple[int, int], int]
MergeRules: TypeAlias = list[MergeRule]
TokenIds: TypeAlias = list[int]


class BPETokenizer:
    def __init__(self):
        self.merge_rules: MergeRules = []
        self.vocab: dict[int, bytes] = {}

    def train(self, text: str, vocab_size: int) -> None:
        result = self.train_bpe(text, vocab_size)
        self.merge_rules = result["merge_rules"]
        self.vocab = result["vocab"]

    def encode(self, text: str) -> list[int]:
        return self._encode(text, self.merge_rules)

    def decode(self, token_ids: TokenIds) -> str:
        return self._decode(token_ids, self.vocab)

    BASE_VOCAB_SIZE = 256

    @staticmethod
    def _text_to_bytes(text: str) -> list[int]:
        """Convert a string to a list of byte values (0-255)"""
        return list(text.encode("utf-8"))

    @staticmethod
    def _get_pair_counts(token_ids: TokenIds) -> Counter:
        """Count how often each adjacent pair appears"""
        counts = Counter()
        for i in range(len(token_ids) - 1):
            pair = (token_ids[i], token_ids[i + 1])
            counts[pair] += 1
        return counts

    @staticmethod
    def _merge(token_ids: TokenIds, pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace all occurrences of `pair` in `token_ids` with `new_id`"""
        result = []
        i = 0
        while i < len(token_ids):
            # Check if this position matches the pair
            if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(token_ids[i])
                i += 1
        return result

    def train_bpe(self, text: str, vocab_size: int) -> dict:
        """
        Train a BPE tokenizer
        """
        msg = f"Base vocab size is {self.BASE_VOCAB_SIZE}, so target vocab size must be greater than {self.BASE_VOCAB_SIZE}"
        assert vocab_size > self.BASE_VOCAB_SIZE, msg

        # TODO: Is there a way to separate out the "Corpus to Pair Counts"
        # processing so that it can be done separately over a series of
        # documents. And then that pre-built Pair Count data can be used as in
        # input for the second phase of training the BPE.

        token_ids = self._text_to_bytes(text)
        num_merges = vocab_size - 256

        # Ordered list of merge rules
        # We want to know what order the rules were "discovered" in so that we
        # can continue to apply them in that way. They are applied based on
        # frequency in the corpus.
        #
        # List of tuples:
        # 1. Pair of consecutive bytes
        # 2. New byte ID to replace pair
        merge_rules: MergeRules = []

        for i in range(num_merges):
            counts = self._get_pair_counts(token_ids)

            # Pick most frequent pair
            next_pair = counts.most_common(1)[0][0]
            new_id = self.BASE_VOCAB_SIZE + i

            token_ids = self._merge(token_ids, next_pair, new_id)
            merge_rules.append((next_pair, new_id))

            if (i + 1) % 50 == 0 or i < 5:
                print(
                    f"Merge {i+1}/{num_merges}: {next_pair} -> {new_id} (count: {counts[next_pair]})"
                )

        # Build vocabulary: base encoding + multi-byte phrases
        vocab = {i: bytes([i]) for i in range(self.BASE_VOCAB_SIZE)}
        for (pair_a, pair_b), new_id in merge_rules:
            print(f"vocab: {new_id} -> {vocab[pair_a]} + {vocab[pair_b]}")
            vocab[new_id] = vocab[pair_a] + vocab[pair_b]

        return {"merge_rules": merge_rules, "vocab": vocab}

    def _encode(self, text: str, merge_rules: MergeRules) -> TokenIds:
        """Encode a string into token IDs using trained merge rules"""
        token_ids = self._text_to_bytes(text)
        for byte_pair, new_id in merge_rules:
            token_ids = self._merge(token_ids, byte_pair, new_id)

        return token_ids

    def _decode(self, token_ids: list[int], vocab: dict[int, bytes]) -> str:
        text_as_bytes = b"".join(vocab[id] for id in token_ids)
        text = text_as_bytes.decode("utf-8", errors="replace")
        return text


def main():
    # Read in a corpus from some source file
    corpus = ""
    file_path = Path("~/dev/jbranchaud/til/combined.md").expanduser()
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            corpus = file.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Pick a vocab size:
    # - 300 is quick
    # - 500 produces more interesting tokens
    vocab_size = 300

    # Train the corpus
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size)
    vocab = tokenizer.vocab

    # Encode something
    test_text = "One of my favorite features that many REPLs offer is an `edit` (Ruby's IRB) or `\edit` (Postgres' psql) command that pops you into your default editor (nvim for me) where you can prepare your statement (or multiple statements!) without worrying about hitting `<enter>` too soon."
    token_ids = tokenizer.encode(test_text)
    print(f"\nEncoded '{test_text}': {token_ids}")
    print(
        f"Token count: {len(test_text.encode('utf-8'))} bytes -> {len(token_ids)} tokens"
    )

    # Decode it
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: '{decoded}'")
    assert decoded == test_text, "Round-trip failed!"

    # Look at the vocab
    print("\nLearned tokens (beyond single bytes):")
    for token_id in range(256, max(vocab.keys()) + 1):
        if token_id in vocab:
            token_as_text = vocab[token_id].decode("utf-8", errors="replace")
            print(f"  {token_id}: {repr(token_as_text)}")


if __name__ == "__main__":
    main()
