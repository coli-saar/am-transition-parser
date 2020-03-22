import torch
from allennlp.common import Registrable
from allennlp.nn.util import masked_log_softmax


class EdgeExistenceLoss(Registrable):

    def compute_loss(self, edge_scores : torch.Tensor, target_gold_edges : torch.Tensor, current_mask : torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes edge existence loss
        :param edge_scores: unnormalized edge scores of shape (batch_size, input_seq_len)
        :param target_gold_edges: for each batch element points to the correct edge, shape (batch_size)
        :param current_mask: shape (batch_size,) denotes for each batch element if we are already in the padding region.
        :param input_mask: shape (batch_size, input_seq_len) denoting padded elements in the input sequence
        :return: tensor of shape (batch_size,) with loss for decision.
        """
        raise NotImplementedError()


@EdgeExistenceLoss.register("nll")
class NLLExistenceLoss(EdgeExistenceLoss):
    """
    Negative log-likelihood existence loss.
    """

    def compute_loss(self, edge_scores : torch.Tensor, target_gold_edges : torch.Tensor, current_mask : torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:

        #compute softmax
        edge_log_softmax = masked_log_softmax(edge_scores, input_mask, dim=1)

        batch_size = edge_log_softmax.shape[0]

        return current_mask * edge_log_softmax[range(batch_size), target_gold_edges]


