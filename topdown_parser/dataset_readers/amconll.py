from typing import Dict, Tuple, List, Any, Iterable
import logging

from allenpipeline import OrderedDatasetReader
from overrides import overrides
from conllu.parser import parse_line, DEFAULT_FIELDS

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, ArrayField, ListField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

from .amconll_tools import parse_amconll, AMSentence
from ..transition_system import TransitionSystem

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

import numpy as np



@DatasetReader.register("amconll")
class AMConllDatasetReader(OrderedDatasetReader):
    """
    Reads a file in amconll format containing AM dependency trees.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    """
    def __init__(self,
                 transition_system : TransitionSystem,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 validate : bool = True,
                 lazy: bool = False, fraction:float = 1.0, only_read_fraction_if_train_in_filename : bool = False) -> None:
        super().__init__(lazy)
        self.validate = validate
        self.transition_system = transition_system
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.fraction = fraction
        self.only_read_fraction_if_train_in_filename = only_read_fraction_if_train_in_filename

    def read_file(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        if self.fraction < 0.9999 and (not self.only_read_fraction_if_train_in_filename or (self.only_read_fraction_if_train_in_filename and "train" in file_path)):
            with open(file_path, 'r') as amconll_file:
                logger.info("Reading a fraction of "+str(self.fraction)+" of the AM dependency trees from amconll dataset at: %s", file_path)
                sents = list(parse_amconll(amconll_file))
                for i,am_sentence in  enumerate(sents):
                    if i <= len(sents) * self.fraction:
                        yield self.text_to_instance(am_sentence)
        else:
            with open(file_path, 'r') as amconll_file:
                logger.info("Reading AM dependency trees from amconll dataset at: %s", file_path)
                for i,am_sentence in  enumerate(parse_amconll(amconll_file)):
                    yield self.text_to_instance(am_sentence)


    @overrides
    def text_to_instance(self,  # type: ignore
                         am_sentence: AMSentence) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        position_in_corpus : ``int``, required.
            The index of this sentence in the corpus.
        am_sentence : ``AMSentence``, required.
            The words in the sentence to be encoded.

        Returns
        -------
        An instance containing words, pos tags, dependency edge labels, head
        indices, supertags and lexical labels as fields.
        """
        fields: Dict[str, Field] = {}

        formalism = am_sentence.attributes["framework"]
        tokens = TextField([Token(w) for w in am_sentence.get_tokens(shadow_art_root=True)], self._token_indexers)
        fields["words"] = tokens
        fields["pos_tags"] = SequenceLabelField(am_sentence.get_pos(), tokens, label_namespace="pos")
        fields["ner_tags"] = SequenceLabelField(am_sentence.get_ner(), tokens, label_namespace="ner_labels")
        fields["lemmas"] = SequenceLabelField(am_sentence.get_lemmas(), tokens, label_namespace="lemmas")

        decisions = list(self.transition_system.get_order(am_sentence))

        if self.validate:
            reconstructed = self.transition_system.get_dep_tree(decisions, am_sentence)
            assert am_sentence == reconstructed

        seq = ListField([ LabelField(decision.position, skip_indexing=True) for decision in decisions])
        fields["seq"] = seq
        fields["active_nodes"] = ListField([ LabelField(active_node, skip_indexing=True) for active_node in self.transition_system.get_active_nodes(am_sentence)])
        fields["labels"] = SequenceLabelField([decision.label for decision in decisions],seq, label_namespace=formalism+"_labels")
        fields["label_mask"] = SequenceLabelField([int(decision.label != "") for decision in decisions],seq)

        # fields["supertags"] = SequenceLabelField(am_sentence.get_supertags(), tokens, label_namespace=formalism+"_supertag_labels")
        # fields["lexlabels"] = SequenceLabelField(am_sentence.get_lexlabels(), tokens, label_namespace=formalism+"_lex_labels")
        # fields["head_tags"] = SequenceLabelField(am_sentence.get_edge_labels(),tokens, label_namespace=formalism+"_head_tags") #edge labels
        # fields["head_indices"] = SequenceLabelField(am_sentence.get_heads(),tokens,label_namespace="head_index_tags")

        fields["metadata"] = MetadataField({"formalism": formalism,
                                            "am_sentence" : am_sentence,
                                            "is_annotated" : am_sentence.is_annotated()})
        return Instance(fields)
