from contextlib import suppress
from chapter_02.bpe_tokenizer import BPETokenizer, BPEConfig
import pytest

def get_test_corpus():
    return """
    I'm a very routine-oriented person, so having this habit of writing TILs has been very anchoring for me. Like any software career I've experienced the ebbs and flows of energy and excitement around both paid work and side-projects. Even when a work project was in a bit of a doldrum or there were challenging team dynamics, I was always able put on a learning mindset. At the end of my day I could look through my notes, see some of the interesting things that had caught my eye, dig in a bit further, and then write a TIL.

    When I first started writing TILs, I was working at a small Ruby on Rails consultancy. I was surrounded by the smartest and kindest software devs I've ever had the privilege to work with. It was intimidating. I was learning a ton every day to the point of overwhelm and yet also felt constantly like I didn't know enough. Writing TILs was as much a way to convince myself that I was steadily improving as anything else.
    """

def test_train_bpe():
    extra_vocab_size = 3
    config = BPEConfig(BPEConfig.BASE_VOCAB_SIZE + extra_vocab_size, [])
    tokenizer = BPETokenizer(config)

    text = get_test_corpus()
    bpe = tokenizer.train_bpe(text)

    assert len(bpe.merge_rules) == extra_vocab_size

    expected_merge_rules = [
        { 'sequence': (32, 97), 'new_id': 256, 'vocab_entries': (b' ', b'a') },
        { 'sequence': (105, 110) , 'new_id': 257, 'vocab_entries': (b'i', b'n') },
        { 'sequence': (32, 119) , 'new_id': 258, 'vocab_entries': (b' ', b'w') }
    ]

    for i, merge_rule in enumerate(bpe.merge_rules):
        expected = expected_merge_rules[i]

        assert expected['sequence'] == merge_rule[0]
        assert expected['new_id'] == merge_rule[1]
        actual_vocab_entries = tuple([bpe.vocab[id] for id in merge_rule[0]])
        assert expected['vocab_entries'] == actual_vocab_entries


def test_bpe_config_with_invalid_vocab_size():
    with pytest.raises(ValueError) as exception:
        BPEConfig(22, [])

    assert "vocab_size (22) must be greater than" in str(exception.value)


def test_train_bpe_with_invalid_vocab_size():
    config = BPEConfig(300, [])
    tokenizer = BPETokenizer(config)

    config.vocab_size = 22

    with pytest.raises(AssertionError) as exception:
        tokenizer.train_bpe("This isn't gonna work.")

    assert "vocab_size (22) must be greater than" in str(exception.value)

def test_merge_with_byte_pair():
    merged_tokens = BPETokenizer._merge([1, 2, 3, 4, 5, 2, 3, 1], (2, 3), 256)
    assert merged_tokens == [1, 256, 4, 5, 256, 1]


def test_merge_with_byte_sequence():
    token_ids = [1, 2, 3, 4, 5, 2, 3, 1, 2, 3, 4, 1]
    merged_tokens = BPETokenizer._merge(token_ids, (2, 3, 4), 256)
    assert merged_tokens == [1, 256, 5, 2, 3, 1, 256, 1]


def test_subsequence_at_index():
    token_ids = [1, 2, 3, 4, 5]
    assert BPETokenizer._subsequence_at_index(token_ids, [3, 4], 2)
    assert not BPETokenizer._subsequence_at_index(token_ids, [3, 4], 1)
