import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Attention
from allennlp.nn.util import masked_log_softmax


class EdgeModel(Model):

    def __init__(self, vocab: Vocabulary):
        super().__init__(vocab)

    def set_input(self, encoded_input : torch.Tensor, mask : torch.Tensor) -> None:
        """
        Set input for current batch
        :param mask: shape (batch_size, input_seq_len)
        :param encoded_input: shape (batch_size, input_seq_len, encoder output dim)
        :return:
        """
        raise NotImplementedError()

    def edge_scores(self, decoder : torch.Tensor) -> torch.Tensor:
        """
        Obtain edge existence scores
        :param decoder: shape (batch_size, decoder dim)
        :return: a tensor of shape (batch_size, input_seq_len) with log-probabilites, normalized over second dimension
        """
        raise NotImplementedError()


@EdgeModel.register("attention")
class AttentionEdgeModel(EdgeModel):
    """
    Wrapper for a simple attention edge model.
    """

    def __init__(self, vocab: Vocabulary, attention : Attention):
        super().__init__(vocab)
        self.attention = attention
        self.attention._normalize = False

        self.encoded_input = None
        self.mask = None

    def set_input(self, encoded_input : torch.Tensor, mask : torch.Tensor) -> None:
        self.encoded_input = encoded_input
        self.mask = mask

    def edge_scores(self, decoder : torch.Tensor) -> torch.Tensor:
        scores = self.attention(decoder, self.encoded_input, self.mask) # (batch_size, input_seq_len)
        assert scores.shape == self.encoded_input.shape[:2]

        return masked_log_softmax(scores, self.mask, dim=1)