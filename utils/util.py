import json
import itertools
from collections import defaultdict, Counter
import numpy as np
from nltk import word_tokenize
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import os
import pickle


def preprocess(data):
    """
    Args:
        data (str):
    Returns: a list of tokens
    """
    tokens = word_tokenize(data)
    return tokens

def process_data(questions_file_name,key_file_name,save_file_name):
    with open(questions_file_name, 'rb') as f:
        data = json.load(f)
    with open(key_file_name, 'rb') as f:
        labels = json.load(f)
    stitched_data = []
    artical_list = list(data['data'].keys())
    for artical_idx in artical_list:
        artical = data['data'][artical_idx]
        context = artical['context']
        question_list = artical['questions'].keys()
        for quiz_idx in question_list:
            quiz = artical['questions'][quiz_idx]
            q = quiz['question']
            option_0 = quiz['answers']['0']
            option_1 = quiz['answers']['1']
            option_2 = quiz['answers']['2']
            option_3 = quiz['answers']['3']
            label = '<' + labels['data'][quiz_idx] + '>'
            paragraph = preprocess(context)
            q_and_a =  ['<q>'] + preprocess(q) + ['<0>'] + preprocess(option_0) + \
                       ['<1>'] + preprocess(option_1) + ['<2>'] + preprocess(option_2) + ['<3>'] + preprocess(option_3)
            stitched_data.append([[paragraph,q_and_a],label])
    with open(save_file_name, 'wb') as f:
        pickle.dump(stitched_data, f)

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
    def __init__(self, texts, labels, vocab=None, labels_vocab=None, parag_max_len=40, q_and_a_max_len=40, lowercase=True):
        """
        Args:
            texts (list): tokenized inputs, with format [paragraph, q_and_a]
            labels (list of str): the correponding labels of the dataset examples
            vocab (MyVocabulary, optional): vocabular to convert text to indices. If not provided, will be created based on the texts
            labels_vocab (MyVocabulary, optional): vocabular to convert labels to indices. If not provided, will be created based on the labels
            parag_max_len (int): maximum length of the paragraph. Texts shorter than parag_max_len will be cut at the end
            q_and_a_max_len (int): maximum length of the q_and_a(question and answers combination). Texts shorter than q_and_a_max_len will be cut at the end
            lowercase (bool, optional): a fag specifying whether or not the input text should be lowercased
        """

        self.parag_max_len = parag_max_len
        self.q_and_a_max_len = q_and_a_max_len
        self.lowercase = lowercase

        self.parag = [self._pad(parag,is_parag=True) for parag,q_and_a in texts]
        self.q_and_a = [self._pad(q_and_a,is_parag=False) for parag,q_and_a in texts]
        self.labels = labels

        if vocab is None:
            vocab = Vocabulary(['<PAD>', '<UNK>', '<q>', '<0>', '<1>', '<2>', '<3>'])
            vocab.add_tokens(itertools.chain.from_iterable(self.parag))
            vocab.add_tokens(itertools.chain.from_iterable(self.q_and_a))

        if labels_vocab is None:
            labels_vocab = Vocabulary()
            labels_vocab.add_tokens(self.labels)

        self.vocab = vocab
        self.labels_vocab = labels_vocab

    def _pad(self, tokens, is_parag=True):
        """
        Pad tokens to self.max_len
        Args:
            tokens (list): a list of str tokens for a given example

        Returns:
            list: a padded list of str tokens for a given example
        """
        # pad the list of tokens to be exactly of the `max_len` size
        ### YOUR CODE BELOW ###
        if is_parag:
            max_len = self.parag_max_len
        else:
            max_len = self.q_and_a_max_len
        if len(tokens) >= max_len:
            tokens = tokens[:max_len]
        else:
            pad_num = max_len - len(tokens)
            tokens = tokens + pad_num * ['<PAD>']
        ### YOUR CODE ABOVE ###
        return tokens

    def __getitem__(self, idx):
        """
        Given an index, return a indexed dataset example

        Args:
            idx (int): dataset index

        Returns:
            tuple: a tuple of token_ids based on the vocabulary mapping  and a corresponding label
        """
        ### YOUR CODE BELOW ###
        parag, q_and_a = self.parag[idx], self.q_and_a[idx]
        parag_tokens = np.array([self.vocab[token] if token in self.vocab else self.vocab['<UNK>'] for token in parag],
                          dtype=np.int64)
        q_and_a_tokens = np.array([self.vocab[token] if token in self.vocab else self.vocab['<UNK>'] for token in q_and_a],
                          dtype=np.int64)
        label = self.labels[idx]
        label = self.labels_vocab[label]
        ### YOUR CODE ABOVE ###
        return parag_tokens, q_and_a_tokens, label

    def __len__(self):
        return len(self.parag)


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.
    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.
    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs

def torch_from_json(path, dtype=torch.float32):
    """Load a PyTorch Tensor from a JSON file.
    Args:
        path (str): Path to the JSON file to load.
        dtype (torch.dtype): Data type of loaded array.
    Returns:
        tensor (torch.Tensor): Tensor loaded from JSON file.
    """
    with open(path, 'r') as fh:
        array = np.array(json.load(fh))

    tensor = torch.from_numpy(array).type(dtype)

    return tensor


def get_dataset():
    """
    The structure of quAIL training dataset:

    {'version': <str>,
    'data':
        'u001':
            'author': <str>
            'title': <str>
            'context': <str>
            'questions':
                'u001_0':
                    'question': <str>
                    'answers':
                        '0': <str>
                        '1': <str>
                        '2': <str>
                        '3': <str>}
    """
    train_processed_data_file_name = 'train_processed_data.txt'
    dev_processed_data_file_name = 'dev_processed_data.txt'

    # create stitched data if they do not exist.
    f_list = os.listdir()
    if train_processed_data_file_name not in f_list:
        questions_file_name = './data/quAIL/train_questions.json'
        key_file_name = './data/quAIL/train_key.json'
        process_data(questions_file_name, key_file_name, train_processed_data_file_name)
    if dev_processed_data_file_name not in f_list:
        questions_file_name = './data/quAIL/dev_questions.json'
        key_file_name = './data/quAIL/new_dev_key.json'
        process_data(questions_file_name, key_file_name, dev_processed_data_file_name)

    # load stitched data
    with open(train_processed_data_file_name, 'rb') as f:
        train_data = pickle.load(f)
    with open(dev_processed_data_file_name, 'rb') as f:
        dev_data = pickle.load(f)

    parag_length_list = [len(example[0]) for example, label in train_data]
    max_parag_length = max(parag_length_list)
    q_and_a_length_list = [len(example[1]) for example, label in train_data]
    max_q_and_a_length = max(q_and_a_length_list)

    train_texts = [example for example, label in train_data]
    train_labels = [label for quiz, label in train_data]
    dev_texts = [quiz for quiz, label in dev_data]
    dev_labels = [label for quiz, label in dev_data]

    # build standard Pytorch Dataset object of our data
    dataset_train = QADataset(train_texts, train_labels, parag_max_len=max_parag_length,
                              q_and_a_max_len=max_q_and_a_length)
    dataset_dev = QADataset(dev_texts, dev_labels, vocab=dataset_train.vocab, labels_vocab=dataset_train.labels_vocab,
                            parag_max_len=max_parag_length, q_and_a_max_len=max_q_and_a_length)


    return dataset_train, dataset_dev


class EMA:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


def load_model(model, checkpoint_path, device, return_step=True):
    """Load model parameters from disk.
    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.
    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    ckpt_dict = torch.load(checkpoint_path, map_location=device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model