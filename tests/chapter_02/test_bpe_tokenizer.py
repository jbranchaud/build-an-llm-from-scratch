from chapter_02.bpe_tokenizer import BPETokenizer


def test_merge_with_byte_pair():
    merged_tokens = BPETokenizer._merge([1, 2, 3, 4, 5, 2, 3, 1], [2, 3], 256)
    assert merged_tokens == [1, 256, 4, 5, 256, 1]


def test_merge_with_byte_sequence():
    token_ids = [1, 2, 3, 4, 5, 2, 3, 1, 2, 3, 4, 1]
    merged_tokens = BPETokenizer._merge(token_ids, [2, 3, 4], 256)
    assert merged_tokens == [1, 256, 5, 2, 3, 1, 256, 1]


def test_subsequence_at_index():
    token_ids = [1, 2, 3, 4, 5]
    assert BPETokenizer._subsequence_at_index(token_ids, [3, 4], 2)
    assert not BPETokenizer._subsequence_at_index(token_ids, [3, 4], 1)
