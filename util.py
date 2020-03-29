import string
import itertools
from collections import defaultdict, Counter
import numpy as np
from nltk import word_tokenize
from torch.utils.data import Dataset, DataLoader

def preprocess(data):
    """
    Args:
        data (str):
    Returns: a list of tokens
    """
    tokens = word_tokenize(data)
    clean_tokens = []
    for token in tokens:
        if token not in string.punctuation:
            clean_tokens.append(token)
    return clean_tokens

class Vocabulary:
    def __init__(self, special_tokens=None):
        self.w2idx = {}
        self.idx2w = {}
        self.w2cnt = defaultdict(int)
        self.special_tokens = special_tokens
        if self.special_tokens is not None:
            self.add_tokens(special_tokens)

    def add_tokens(self, tokens):
        for token in tokens:
            self.add_token(token)
            self.w2cnt[token] += 1

    def add_token(self, token):
        if token not in self.w2idx:
            cur_len = len(self)
            self.w2idx[token] = cur_len
            self.idx2w[cur_len] = token

    def prune(self, min_cnt=2):
        to_remove = set([token for token in self.w2idx if self.w2cnt[token] < min_cnt])
        if self.special_tokens is not None:
            to_remove = to_remove.difference(set(self.special_tokens))
        for token in to_remove:
            self.w2cnt.pop(token)
        self.w2idx = {token: idx for idx, token in enumerate(self.w2cnt.keys())}
        self.idx2w = {idx: token for token, idx in self.w2idx.items()}

    def __contains__(self, item):
        return item in self.w2idx

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.w2idx[item]
        elif isinstance(item, int):
            return self.idx2w[item]
        else:
            raise TypeError("Supported indices are int and str")

    def __len__(self):
        return (len(self.w2idx))


class QADataset(Dataset):
    def __init__(self, texts, labels, vocab=None, labels_vocab=None, max_len=40, lowercase=True):
        """
        Args:
            texts (list of list of str): texts of the dataset examples, one example could be [context, question, option1, option2, option3, option4]
            labels (list of str): the correponding labels of the dataset examples
            vocab (MyVocabulary, optional): vocabular to convert text to indices. If not provided, will be created based on the texts
            labels_vocab (MyVocabulary, optional): vocabular to convert labels to indices. If not provided, will be created based on the labels
            max_len (int): maximum length of the text. Texts shorter than max_len will be cut at the end
            lowercase (bool, optional): a fag specifying whether or not the input text should be lowercased
        """

        self.max_len = max_len
        self.lowercase = lowercase

        self.texts = [self._preprocess(t, max_len=max_len, lowercase=lowercase) for t in texts]
        self.labels = labels

        if vocab is None:
            vocab = Vocabulary(['<PAD>', '<UNK>'])
            vocab.add_tokens(itertools.chain.from_iterable(self.texts))

        if labels_vocab is None:
            labels_vocab = Vocabulary()
            labels_vocab.add_tokens(labels)

        self.vocab = vocab
        self.labels_vocab = labels_vocab


def _preprocess(self, text, max_len=None, lowercase=True):
    """
    Preprocess a give dataset example
    Args:
        text (str): given dataset example
        max_len (int, optional): maximum sequence length
        lowercase (bool, optional): a fag specifying whether or not the input text should be lowercased

    Returns:
        a list of tokens for a given text span
    """
    # tokenize the input text
    ### YOUR CODE BELOW ###
    if lowercase:
        text = text.lower()
    tokens = word_tokenize(text)
    # cut the list of tokens to `max_len` if needed
    tokens = self._pad(tokens)
    ### YOUR CODE ABOVE ###
    return tokens


def _pad(self, tokens):
    """
    Pad tokens to self.max_len
    Args:
        tokens (list): a list of str tokens for a given example

    Returns:
        list: a padded list of str tokens for a given example
    """
    # pad the list of tokens to be exactly of the `max_len` size
    ### YOUR CODE BELOW ###
    max_len = self.max_len
    if len(tokens) >= max_len:
        tokens = tokens[:max_len]
    else:
        pad_num = max_len - len(tokens)
        tokens = tokens + pad_num * ['<PAD>']
    ### YOUR CODE ABOVE ###
    return tokens

def __getitem__(self, idx):
    """
    Given an index, return a formatted dataset example

    Args:
        idx (int): dataset index

    Returns:
        tuple: a tuple of token_ids based on the vocabulary mapping  and a corresponding label
    """
    ### YOUR CODE BELOW ###
    tokens = self.texts[idx]
    tokens = np.array([self.vocab[token] if token in self.vocab else self.vocab['<UNK>'] for token in tokens],
                      dtype=np.int64)
    label = self.labels[idx]
    label = self.labels_vocab[label]
    ### YOUR CODE ABOVE ###
    return tokens, label

def __len__(self):
    return len(self.texts)