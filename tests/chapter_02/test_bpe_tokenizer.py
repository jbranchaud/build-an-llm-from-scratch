from chapter_02.bpe_tokenizer import BPETokenizer
import pytest

def get_test_corpus():
    return """
    I'm a very routine-oriented person, so having this habit of writing TILs has been very anchoring for me. Like any software career I've experienced the ebbs and flows of energy and excitement around both paid work and side-projects. Even when a work project was in a bit of a doldrum or there were challenging team dynamics, I was always able put on a learning mindset. At the end of my day I could look through my notes, see some of the interesting things that had caught my eye, dig in a bit further, and then write a TIL.

    When I first started writing TILs, I was working at a small Ruby on Rails consultancy. I was surrounded by the smartest and kindest software devs I've ever had the privilege to work with. It was intimidating. I was learning a ton every day to the point of overwhelm and yet also felt constantly like I didn't know enough. Writing TILs was as much a way to convince myself that I was steadily improving as anything else.
    """

def test_train_bpe():
    tokenizer = BPETokenizer()

    text = get_test_corpus()
    bpe = tokenizer.train_bpe(text, BPETokenizer.BASE_VOCAB_SIZE + 1, [])

    assert "merge_rules" in bpe
    assert "vocab" in bpe

    last_merge_rule = bpe['merge_rules'][-1]
    assert last_merge_rule[0] == (32, 97)
    assert last_merge_rule[1] == 256

    vocab = bpe["vocab"]
    assert vocab[last_merge_rule[0][0]] == b' '
    assert vocab[last_merge_rule[0][1]] == b'a'



def test_train_bpe_with_invalid_vocab_size():
    tokenizer = BPETokenizer()

    with pytest.raises(AssertionError) as exception:
        tokenizer.train_bpe("This is the corpus", 22, [])

    assert "target vocab size must be greater than" in str(exception.value)

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
