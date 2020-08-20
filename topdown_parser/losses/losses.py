import torch
from allennlp.common import Registrable


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.

    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.
        vector = vector + (mask + 1e-30).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


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


