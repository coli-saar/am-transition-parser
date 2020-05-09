import torch
from allennlp.common import Registrable
from allennlp.nn import Activation
from torch.nn import Module

import numpy as np

@Registrable.register("kg_edges")
class KGEdges(Module, Registrable):
    """
    Reimplementation of the edge model of the graph-based graph_dependency_parser by Kiperwasser and Goldberg (2016): https://aclweb.org/anthology/Q16-1023
    """
    def __init__(self,
                 encoder_dim:int,
                 edge_dim: int,
                 activation : Activation = None) -> None:
        """
            Parameters
            ----------
            vocab : ``Vocabulary``, required
                A Vocabulary, required in order to compute sizes for input/output projections.
            encoder_dim : ``int``, required.
                The output dimension of the encoder.
            label_dim : ``int``, required.
                The dimension of the hidden layer of the MLP used for predicting the edge labels.
            edge_dim : ``int``, required.
                The dimension of the hidden layer of the MLP used for predicting edge existence.
            edge_label_namespace: str,
                The namespace of the edge labels: a combination of the task name + _head_tags
            activation : ``Activation``, optional, (default = tanh).
                The activation function used in the MLPs.
            dropout : ``float``, optional, (default = 0.0)
                The variational dropout applied to the output of the encoder and MLP layers.
        """
        super(KGEdges, self).__init__()
        self._encoder_dim = encoder_dim
        if activation is None:
            self.activation = Activation.by_name("tanh")()
        else:
            self.activation = activation

        #edge existence:

        #these two matrices together form the feed forward network which takes the vectors of the two words in question and makes predictions from that
        #this is the trick described by Kiperwasser and Goldberg to make training faster.
        self.head_arc_feedforward = torch.nn.Linear(encoder_dim, edge_dim)
        self.child_arc_feedforward = torch.nn.Linear(encoder_dim, edge_dim, bias=False) #bias is already added by head_arc_feedforward

        self.arc_out_layer = torch.nn.Linear(edge_dim, 1, bias=False)  # K&G don't use a bias for the output layer

    def encoder_dim(self):
        return self._encoder_dim

    def edge_existence(self,encoded_text: torch.Tensor, mask : torch.LongTensor) -> torch.Tensor:
        """
        Computes edge existence scores for a batch of sentences.

        Parameters
        ----------
        encoded_text : torch.Tensor, required
            The input sentence, with artificial root node (head sentinel) added in the beginning of
            shape (batch_size, sequence length, encoding dim)
        mask : ``torch.LongTensor``
            A mask denoting the padded elements in the batch.

        Returns
        -------
        attended_arcs: torch.Tensor
            The edge existence scores in a tensor of shape (batch_size, sequence_length, sequence_length). The mask is taken into account.
        """
        float_mask = mask.float()

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self.head_arc_feedforward(encoded_text)
        child_arc_representation = self.child_arc_feedforward(encoded_text)

        bs,sl,arc_dim = head_arc_representation.size()

        #now repeat the token representations to form a matrix:
        #shape (batch_size, sequence_length, sequence_length, arc_representation_dim)
        heads = head_arc_representation.repeat(1,sl,1).reshape(bs,sl,sl,arc_dim) #heads in one direction
        deps = child_arc_representation.repeat(1, sl, 1).reshape(bs, sl, sl, arc_dim).transpose(1,2) #deps in the other direction

        # shape (batch_size, sequence_length, sequence_length, arc_representation_dim)
        combined = self.activation(heads + deps) #now the feedforward layer that takes every pair of vectors for tokens is complete.
        #combined now represents the activations in the hidden layer of the MLP.
        edge_scores = self.arc_out_layer(combined).squeeze(3) #last dimension is now 1, remove it

        #mask out stuff for padded tokens:
        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        edge_scores = edge_scores + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)
        return edge_scores

    def greedy_decode_arcs(self,
                                existence_scores: torch.Tensor,
                                mask: torch.Tensor) -> torch.Tensor:
        """
        Decodes the head  predictions by decoding the unlabeled arcs
        independently for each word. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).

        Parameters
        ----------
        existence_scores : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        mask: torch.Tensor, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        existence_scores = existence_scores + torch.diag(existence_scores.new(mask.size(1)).fill_(-np.inf))
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = (1 - mask).bool().unsqueeze(2)
            existence_scores.masked_fill_(minus_mask, -np.inf)

        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = existence_scores.max(dim=2)
        return heads