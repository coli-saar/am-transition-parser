import math
from copy import deepcopy

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward
from torch.nn import Parameter
import torch.nn.functional as F


class EdgeLabelModel(Model):

    def __init__(self, vocab: Vocabulary, formalism : str):
        super().__init__(vocab)
        self.formalism = formalism
        self.output_namespace = formalism+"_labels"
        self.vocab_size = self.vocab.get_vocab_size(self.output_namespace)

    def set_input(self, encoded_input : torch.Tensor, mask : torch.Tensor) -> None:
        """
        Set input for current batch.
        :param mask: shape (batch_size, input_seq_len)
        :param encoded_input: shape (batch_size, input_seq_len, encoder output dim)
        :return:
        """
        raise NotImplementedError()

    def edge_label_scores(self, encoder_indices : torch.Tensor, decoder : torch.Tensor) -> torch.Tensor:
        """
        Retrieve label scores for decoder vector and indices into the encoded_input.
        :param encoder_indices: shape (batch_size) indicating the destination of the edge
        :param decoder: shape (batch_size, decoder_dim)
        :return: a tensor of shape (batch_size, edge_label_vocab) with log probabilities, normalized over vocabulary dimension.
        """
        raise NotImplementedError()


@EdgeLabelModel.register("simple")
class SimpleEdgeLabelModel(EdgeLabelModel):

    def __init__(self, vocab: Vocabulary, formalism : str, mlp : FeedForward):
        super().__init__(vocab,formalism)
        self.feedforward = mlp
        self.output_layer = torch.nn.Linear(mlp.get_output_dim(), self.vocab_size)

    def set_input(self, encoded_input : torch.Tensor, mask : torch.Tensor) -> None:
        self.encoded_input = encoded_input
        self.mask = mask

    def edge_label_scores(self, encoder_indices : torch.Tensor, decoder : torch.Tensor) -> torch.Tensor:
        """

        :param encoder_indices: (batch_size,)
        :param decoder: shape (batch_size, decoder_dim)
        :return:
        """
        batch_size, _, encoder_dim = self.encoded_input.shape
        vectors_in_question = self.encoded_input[range(batch_size),encoder_indices, :]
        assert vectors_in_question.shape == (batch_size, encoder_dim)

        logits = self.output_layer(self.feedforward(torch.cat([vectors_in_question, decoder], dim=1)))

        assert logits.shape == (batch_size, self.vocab_size)

        return torch.log_softmax(logits, dim=1)



# @EdgeLabelModel.register("ma")
# class MaEdgeModel(EdgeLabelModel):
#
#     def __init__(self, vocab: Vocabulary, formalism : str, mlp: FeedForward):
#         super().__init__(vocab, formalism)
#         self.head_mlp = mlp
#         self.dep_mlp = deepcopy(mlp)
#
#         self._weight_matrix = Parameter(torch.Tensor(self.head_mlp.get_output_dim(), self.dep_mlp.get_output_dim()))
#         self._bias = Parameter(torch.zeros(self.vocab_size))
#
#         self.q_weight = Parameter(torch.Tensor(self.head_mlp.get_output_dim()))
#         self.key_weight = Parameter(torch.Tensor(self.dep_mlp.get_output_dim()))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         bound = 1 / math.sqrt(self.vector_dim)
#         torch.nn.init.uniform_(self.q_weight, -bound, bound)
#         bound = 1 / math.sqrt(self.matrix_dim)
#         torch.nn.init.uniform_(self.key_weight, -bound, bound)
#         torch.nn.init.xavier_uniform_(self._weight_matrix)
#         torch.nn.init.constant_(self._bias, 0.)
#
#     def set_input(self, encoded_input: torch.Tensor, mask: torch.Tensor) -> None:
#         # encoded_input: (batch_size, seq_len, encoder_dim)
#         self.batch_size, self.seq_len, _ = encoded_input.shape
#
#         self.dependent_rep = self.dep_mlp(encoded_input) #(batch_size, seq_len, dependent_dim)
#         self.dependent_rep_with_matrix = F.linear(self.dependent_rep, self._weight_matrix)
#         self.mask = mask
#
#     def edge_label_scores(self, encoder_indices : torch.Tensor, decoder : torch.Tensor) -> torch.Tensor:
#
#         batch_size, _, encoder_dim = self.encoded_input.shape
#         vectors_in_question = self.dependent_rep[range(batch_size),encoder_indices, :] #shape (batch_size, vector dim)
#
#         head_rep = self.head_mlp(decoder) #shape (batch_size, decoder dim)
#
#         head_term = torch.einsum("bv, v -> b", head_rep, self.q_weight)
#         dep_term = torch.einsum("bv, v -> v", vectors_in_question, self.key_weight)
#
#         interaction = torch.einsum("", )
#         #shape (batch_size, vocab size)
#         return torch.log_softmax(logits, dim=1)





