import math
from copy import deepcopy

import torch
from allennlp.common import Registrable
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward
from torch.nn import Parameter, Module
import torch.nn.functional as F

from topdown_parser.dataset_readers.AdditionalLexicon import AdditionalLexicon


class EdgeLabelModel(Registrable, Module):

    def __init__(self, lexicon : AdditionalLexicon):
        super().__init__()
        self.lexicon = lexicon
        self.vocab_size = lexicon.vocab_size("edge_labels")

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

    def __init__(self, lexicon : AdditionalLexicon, mlp : FeedForward):
        super().__init__(lexicon)
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


@EdgeLabelModel.register("ma")
class MaEdgeModel(EdgeLabelModel):

    def __init__(self, lexicon : AdditionalLexicon, mlp: FeedForward):
        super().__init__(lexicon)
        self.head_mlp = mlp
        self.dep_mlp = deepcopy(mlp)

        self._U2a = torch.nn.Linear(self.dep_mlp.get_output_dim(), self.vocab_size, bias=False)
        self._U2b = torch.nn.Linear(self.head_mlp.get_output_dim(), self.vocab_size, bias=False)

        self._bilinear = torch.nn.Bilinear(self.head_mlp.get_output_dim(), self.dep_mlp.get_output_dim(), self.vocab_size)


    def set_input(self, encoded_input: torch.Tensor, mask: torch.Tensor) -> None:
        # encoded_input: (batch_size, seq_len, encoder_dim)
        self.batch_size, self.seq_len, _ = encoded_input.shape

        self.dependent_rep = self.dep_mlp(encoded_input) #(batch_size, seq_len, dependent_dim)
        self.dependent_times_matrix = self._U2a(self.dependent_rep)
        self.mask = mask

    def edge_label_scores(self, encoder_indices : torch.Tensor, decoder : torch.Tensor) -> torch.Tensor:

        vectors_in_question = self.dependent_rep[range(self.batch_size),encoder_indices, :] #shape (batch_size, vector dim)
        dependent_with_matrix = self.dependent_times_matrix[range(self.batch_size), encoder_indices,:]

        head_rep = self.head_mlp(decoder) #shape (batch_size, decoder dim)

        logits = self._bilinear(head_rep, vectors_in_question) + self._U2b(head_rep) + dependent_with_matrix
        return torch.log_softmax(logits, dim=1)





