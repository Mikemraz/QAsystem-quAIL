"""Assortment of layers for use in models.py.
Author:
    Chris Chute (chute@stanford.edu)
"""
import math
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.modules import TimeDistributed

from utils.util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.
    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, embedding_size, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.
    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).
    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.
    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.
    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, (h_n,c_n) = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x, h_n


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.
    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).
    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).
        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.
        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class TriLinearAttention(nn.Module):
    """
    This function is taken from Allen NLP group, refer to github:
    https://github.com/chrisc36/allennlp/blob/346e294a5bab1ec0d8f2af962cfe44abc450c369/allennlp/modules/tri_linear_attention.py

    TriLinear attention as used by BiDaF, this is less flexible more memory efficient then
    the `linear` implementation since we do not create a massive
    (batch, context_len, question_len, dim) matrix
    """

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self._x_weights = Parameter(torch.Tensor(input_dim, 1))
        self._y_weights = Parameter(torch.Tensor(input_dim, 1))
        self._dot_weights = Parameter(torch.Tensor(1, 1, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self.input_dim * 3 + 1))
        self._y_weights.data.uniform_(-std, std)
        self._x_weights.data.uniform_(-std, std)
        self._dot_weights.data.uniform_(-std, std)

    def forward(self, matrix_1, matrix_2):
        # pylint: disable=arguments-differ

        # Each matrix is (batch_size, time_i, input_dim)
        batch_dim = matrix_1.shape[0]
        time_1 = matrix_1.shape[1]
        time_2 = matrix_2.shape[1]

        # (batch * time1, dim) * (dim, 1) -> (batch * time1, 1)
        x_factors = torch.matmul(matrix_1.resize(batch_dim * time_1, self.input_dim), self._x_weights)
        x_factors = x_factors.contiguous().view(batch_dim, time_1, 1)  # ->  (batch, time1, 1)

        # (batch * time2, dim) * (dim, 1) -> (batch * tim2, 1)
        y_factors = torch.matmul(matrix_2.resize(batch_dim * time_2, self.input_dim), self._y_weights)
        y_factors = y_factors.contiguous().view(batch_dim, 1, time_2)  # ->  (batch, 1, time2)

        weighted_x = matrix_1 * self._dot_weights  # still (batch, time1, dim)

        matrix_2_t = torch.transpose(matrix_2, 1, 2)  # -> (batch, dim, time2)

        # Batch multiplication,
        # (batch, time1, dim), (batch, dim, time2) -> (batch, time1, time2)
        dot_factors = torch.matmul(weighted_x, matrix_2_t)

        # Broadcasting will correctly repeat the x/y factors as needed,
        # result is (batch, time1, time2)
        return dot_factors + x_factors + y_factors


class SelfAtt(nn.Module):
    """
        The self attention layer is implemented by Gendong Zhang, with the function TimeDistributed provided by Allen NLP.
        The self attention get the attention score of cotext and context.
        Refer to : https://github.com/Oceanland-428/Improved-BiDAF-with-Self-Attention#overview
        Args:
            hidden_size (int): Size of hidden activations.
            drop_prob (float): Probability of zero-ing out activations
    """

    def __init__(self, hidden_size, drop_prob):
        super(SelfAtt, self).__init__()

        self.drop_prob = drop_prob
        self.att_wrapper = TimeDistributed(nn.Linear(hidden_size * 4, hidden_size))
        self.trilinear = TriLinearAttention(hidden_size)
        self.self_att_upsampler = TimeDistributed(nn.Linear(hidden_size * 3, hidden_size * 4))
        self.enc = nn.GRU(hidden_size, hidden_size // 2, 1,
                          batch_first=True,
                          bidirectional=True)
        self.hidden_size = hidden_size

    def forward(self, att, c_mask):
        # (batch_size, c_len, 1600)
        att_copy = att.clone()  # To save the original data of attention from pervious layer.
        # (batch_size * c_len, 1600)
        att_wrapped = self.att_wrapper(
            att)  # unroll the second dimention with the first dimension, and roll it back, change of dimension.
        # non-linearity activation function
        att = F.relu(att_wrapped)  # (batch_size * c_len, 1600)
        #         print("att", att.shape)
        c_mask = c_mask.unsqueeze(dim=2).float()  # (batch_size, c_len, 1)

        drop_att = F.dropout(att, self.drop_prob, self.training)  # (batch_size * c_len, hidden_size)
        #         c_mask = c_mask.permute(1, 0, 2)
        #         print(drop_att.shape, c_mask.shape)

        encoder, _ = self.enc(drop_att)
        #         encoder = self.get_similarity_matrix(drop_att, c_mask)
        #         print("encoder", encoder.shape)
        #         encoder = encoder.unsqueeze(dim=3)

        self_att = self.trilinear(encoder, encoder)  # get the self attention (batch_size, c_len, c_len)

        # to match the shape of the attention matrix
        mask = (c_mask.view(c_mask.shape[0], c_mask.shape[1], 1) * c_mask.view(c_mask.shape[0], 1, c_mask.shape[1]))
        identity = torch.eye(c_mask.shape[1], c_mask.shape[1]).cuda().view(1, c_mask.shape[1], c_mask.shape[1])
        mask = mask * (1 - identity)

        # get the self attention vector features
        self_att_softmax = masked_softmax(self_att, mask, log_softmax=False)
        self_att_vector = torch.matmul(self_att_softmax, encoder)

        # concatenate to make the shape (batch, c_len, 1200)
        conc = torch.cat((self_att_vector, encoder, encoder * self_att_vector), dim=-1)
        #         print("conc", conc.shape)

        # To match with the input attention, we have to upsample the hidden_size from 1200 to 1600.
        upsampler = self.self_att_upsampler(conc)
        out = F.relu(upsampler)

        # (batch_size, c_len, 1600)
        att_copy += out

        att = F.dropout(att_copy, self.drop_prob, self.training)
        return att


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.
    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.
    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    # def __init__(self, hidden_size, parag_length=494, num_cls=4):
    #     super(BiDAFOutput, self).__init__()
    #     self.att_linear = nn.Linear(8 * hidden_size, 1)
    #     self.mod_linear = nn.Linear(2 * hidden_size, 1)
    #
    #     self.linear = nn.Linear(parag_length, num_cls)
    #
    # def forward(self, att, mod):
    #     # att Dim: (batch_size, parag_len, 8*hidden_size)
    #     logits = self.att_linear(att) + self.mod_linear(mod) #dim(batch,parag,1)
    #     logits = logits.squeeze() #dim(batch,parag)
    #     logits = self.linear(logits) #dim(batch,num_cls)
    #     #p = self.softmax(logits)
    #
    #     return logits

    def __init__(self, hidden_size, num_cls=4):
        super(BiDAFOutput, self).__init__()
        self.mod_linear = nn.Linear(hidden_size, num_cls)

    def forward(self, h_n):
        # att Dim: (batch_size, parag_len, 8*hidden_size)
        h_n = h_n.permute(1, 0, 2)
        h_n = torch.sum(h_n,1)
        logits = self.mod_linear(h_n) #dim(batch,num_cls)

        return logits