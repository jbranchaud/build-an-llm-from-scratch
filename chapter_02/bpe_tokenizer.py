import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, TypeAlias
from typing import NewType

ByteSequence = NewType("ByteSequence", tuple[int, ...])

MergeRule: TypeAlias = tuple[ByteSequence, int]
MergeRules: TypeAlias = list[MergeRule]
TokenIds = NewType("TokenIds", list[int])
Vocab: TypeAlias = dict[int, bytes]

@dataclass
class BPEConfig:
    vocab_size: int
    special_tokens: list[str]

    def __post_init__(self):
        if self.vocab_size < BPETokenizer.BASE_VOCAB_SIZE:
            msg = f"vocab_size ({self.vocab_size}) must be greater than or equal to BASE_VOCAB_SIZE ({BPETokenizer.BASE_VOCAB_SIZE})"
            raise ValueError(msg)


@dataclass
class TrainResult:
    merge_rules: MergeRules
    vocab: Vocab


class BPETokenizer:
    def __init__(self, config: BPEConfig) -> None:
        self.merge_rules: MergeRules = []
        self.vocab: Vocab = {}
        self.config: BPEConfig = config

    def train(self, text: str) -> None:
        result = self.train_bpe(text)
        self.merge_rules = result.merge_rules
        self.vocab = result.vocab

    def encode(self, text: str) -> list[int]:
        return self._encode(text, self.merge_rules)

    def decode(self, token_ids: TokenIds) -> str:
        return self._decode(token_ids, self.vocab)

    BASE_VOCAB_SIZE = 256

    @staticmethod
    def _text_to_bytes(text: str) -> TokenIds:
        """Convert a string to a list of byte values (0-255)"""
        return TokenIds(list(text.encode("utf-8")))

    @staticmethod
    def _join_bytes(bytes: Iterable[bytes]) -> bytes:
        return b"".join(bytes)

    @staticmethod
    def _get_pair_counts(token_ids: TokenIds) -> Counter:
        """Count how often each adjacent pair appears"""
        counts = Counter[ByteSequence]()
        for i in range(len(token_ids) - 1):
            pair = ByteSequence((token_ids[i], token_ids[i + 1]))
            counts[pair] += 1
        return counts

    @staticmethod
    def _merge(token_ids: TokenIds, sequence: ByteSequence, new_id: int) -> TokenIds:
        """Replace all occurrences of `pair` in `token_ids` with `new_id`"""
        result = []
        i = 0
        while i < len(token_ids):
            # Check if this position matches the pair
            room_remaining = i < len(token_ids) - (len(sequence) - 1)
            if room_remaining and BPETokenizer._subsequence_at_index(
                token_ids, sequence, i
            ):
                result.append(new_id)
                i += len(sequence)
            else:
                result.append(token_ids[i])
                i += 1
        return TokenIds(result)

    @staticmethod
    def _subsequence_at_index(
        token_ids: TokenIds, sequence: ByteSequence, index: int
    ) -> bool:
        """Check if the sequence appears in token_ids starting at index"""
        for i in range(len(sequence)):
            if sequence[i] != token_ids[index + i]:
                return False

        return True

    def train_bpe(self, text: str) -> TrainResult:
        """
        Train a BPE tokenizer
        """
        msg = f"Target vocab_size ({self.config.vocab_size}) must be greater than BASE_VOCAB_SIZE ({self.BASE_VOCAB_SIZE})"
        assert self.config.vocab_size > self.BASE_VOCAB_SIZE, msg

        # TODO: Is there a way to separate out the "Corpus to Pair Counts"
        # processing so that it can be done separately over a series of
        # documents. And then that pre-built Pair Count data can be used as in
        # input for the second phase of training the BPE.

        token_ids = self._text_to_bytes(text)
        num_merges = self.config.vocab_size - 256

        # Ordered list of merge rules
        # We want to know what order the rules were "discovered" in so that we
        # can continue to apply them in that way. They are applied based on
        # frequency in the corpus.
        #
        # List of tuples:
        # 1. Pair of consecutive bytes
        # 2. New byte ID to replace pair
        merge_rules: MergeRules = []

        # First, apply special tokens to `token_ids` and `merge_rules`
        for special_token in self.config.special_tokens:
            base_tokens_for_special_token = self._text_to_bytes(special_token)

            # TODO: follow template of following section going through
            # `token_ids` and replacing the special token sequence with
            # a `new_id` and then also adding that rule to `merge_rules`.

        for i in range(num_merges):
            counts = self._get_pair_counts(token_ids)

            # Pick most frequent sequence
            next_sequence: ByteSequence = counts.most_common(1)[0][0]
            new_id = self.BASE_VOCAB_SIZE + i

            token_ids = self._merge(token_ids, next_sequence, new_id)
            merge_rules.append((next_sequence, new_id))

            if (i + 1) % 50 == 0 or i < 5:
                print(
                    f"Merge {i+1}/{num_merges}: {next_sequence} -> {new_id} (count: {counts[next_sequence]})"
                )

        # Build vocabulary: base encoding + multi-byte phrases
        vocab = {i: bytes([i]) for i in range(self.BASE_VOCAB_SIZE)}
        for sequence_ids, new_id in merge_rules:
            byte_seq = [vocab[id] for id in sequence_ids]
            bytes_for_print = [f"{byte!r}" for byte in byte_seq]
            print(f"vocab: {new_id} -> {" + ".join(bytes_for_print)}")
            vocab[new_id] = BPETokenizer._join_bytes(byte_seq)

        return TrainResult(merge_rules, vocab)

    def _encode(self, text: str, merge_rules: MergeRules) -> TokenIds:
        """Encode a string into token IDs using trained merge rules"""
        token_ids = self._text_to_bytes(text)
        for byte_pair, new_id in merge_rules:
            token_ids = self._merge(token_ids, byte_pair, new_id)

        return token_ids

    def _decode(self, token_ids: TokenIds, vocab: dict[int, bytes]) -> str:
        text_as_bytes = BPETokenizer._join_bytes(vocab[id] for id in token_ids)
        text = text_as_bytes.decode("utf-8", errors="replace")
        return text


def main(args):
    # Read in a corpus from some source file
    corpus = ""
    try:
        with args.corpus as file:
            corpus = file.read()
    except FileNotFoundError:
        # FIXME: we don't ever get here now because argparse ensures the file exists earlier
        print(f"Error: The file '{args.corpus}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    special_tokens = args.special_tokens or []
    for token in special_tokens:
        print(f"Adding special token: {token}")

    # TODO: remove this once special tokens are implemented
    if len(special_tokens) > 0:
        raise ValueError("Special tokens are not supported yet")

    DEFAULT_VOCAB_SIZE = 300

    # Pick a vocab size:
    # - 300 is quick
    # - 500 produces more interesting tokens
    vocab_size = (args.vocab_size or DEFAULT_VOCAB_SIZE) + len(special_tokens)

    # Train the corpus
    config = BPEConfig(vocab_size, special_tokens)
    tokenizer = BPETokenizer(config)
    tokenizer.train(corpus)
    vocab = tokenizer.vocab

    # Encode something
    test_text = r"One of my favorite features that many REPLs offer is an `edit` (Ruby's IRB) or `\edit` (Postgres' psql) command that pops you into your default editor (nvim for me) where you can prepare your statement (or multiple statements!) without worrying about hitting `<enter>` too soon."
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
    parser = argparse.ArgumentParser(
        description="produce BPE tokenization of corpus of text"
    )
    parser.add_argument(
        "--corpus",
        type=argparse.FileType("r"),
        help="a relative path to a text file",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=argparse.FileType("w"),
        help="a relative path to output BPE representation",
        required=True,
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Override default vocab size of 300, must be greater than 256",
        required=False,
    )
    parser.add_argument(
        "--special-tokens",
        action="extend",
        type=str,
        nargs="+",
        help="List of special tokens to add to the vocab",
        required=False,
    )
    args = parser.parse_args()

    main(args)
