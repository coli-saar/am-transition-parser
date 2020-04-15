import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward
from torch import nn


class Supertagger(Model):

    def __init__(self, vocab: Vocabulary, formalism : str, suffix_namespace : str):
        super().__init__(vocab)
        self.formalism = formalism
        self.output_namespace = formalism+"_" + suffix_namespace
        assert suffix_namespace in ["supertags", "lex_labels"]

        self.vocab_size = self.vocab.get_vocab_size(self.output_namespace)

    def set_input(self, encoded_input: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Set input for current batch
        :param mask: shape (batch_size, input_seq_len)
        :param encoded_input: shape (batch_size, input_seq_len, encoder output dim)
        :return:
        """
        raise NotImplementedError()

    def tag_scores(self, decoder: torch.Tensor, active_node : torch.Tensor) -> torch.Tensor:
        """
        Obtain supertag scores
        :param active_node: shape (batch_size,) with indices to tokens for which we want to choose the supertag.
        :param decoder: shape (batch_size, decoder dim)
        :return: a tensor of shape (batch_size, supertag vocab size) with raw scores for each supertag.
        """
        raise NotImplementedError()


@Supertagger.register("simple-tagger")
class SimpleTagger(Supertagger):

    def __init__(self, vocab: Vocabulary, formalism: str, suffix_namespace: str, mlp : FeedForward):
        super().__init__(vocab, formalism, suffix_namespace)

        self.mlp = mlp
        self.output_layer = nn.Linear(mlp.get_output_dim(), self.vocab_size)

    def set_input(self, encoded_input: torch.Tensor, mask: torch.Tensor) -> None:
        self.encoded_input = encoded_input
        self.mask = mask

    def tag_scores(self, decoder: torch.Tensor, active_node : torch.Tensor) -> torch.Tensor:
        batch_size = active_node.shape[0]

        #Find embeddings of active nodes.
        #relevant_tokens = self.encoded_input[range(batch_size), active_node] #shape (batch_size, encoder dim)

        return self.output_layer(self.mlp(decoder))


@Supertagger.register("combined-tagger")
class CombinedTagger(Supertagger):

    def __init__(self, vocab: Vocabulary, formalism: str, suffix_namespace: str, mlp : FeedForward):
        super().__init__(vocab, formalism, suffix_namespace)

        self.mlp = mlp
        self.output_layer = nn.Linear(mlp.get_output_dim(), self.vocab_size)

    def set_input(self, encoded_input: torch.Tensor, mask: torch.Tensor) -> None:
        self.encoded_input = encoded_input
        self.mask = mask

    def tag_scores(self, decoder: torch.Tensor, active_node : torch.Tensor) -> torch.Tensor:
        batch_size = active_node.shape[0]

        #Find embeddings of active nodes.
        relevant_tokens = self.encoded_input[range(batch_size), active_node] #shape (batch_size, encoder dim)

        return self.output_layer(self.mlp(torch.cat([decoder, relevant_tokens], dim=1)))


@Supertagger.register("no-decoder-tagger")
class CombinedTagger(Supertagger):

    def __init__(self, vocab: Vocabulary, formalism: str, suffix_namespace: str, mlp : FeedForward):
        super().__init__(vocab, formalism, suffix_namespace)

        self.mlp = mlp
        self.output_layer = nn.Linear(mlp.get_output_dim(), self.vocab_size)

    def set_input(self, encoded_input: torch.Tensor, mask: torch.Tensor) -> None:
        self.encoded_input = self.output_layer(self.mlp(encoded_input))
        self.mask = mask

    def tag_scores(self, decoder: torch.Tensor, active_node : torch.Tensor) -> torch.Tensor:
        batch_size = active_node.shape[0]

        #Find embeddings of active nodes.
        return self.encoded_input[range(batch_size), active_node] #shape (batch_size, encoder dim)
