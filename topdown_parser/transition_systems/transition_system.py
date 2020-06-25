from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict, Any, Set, Type

import torch
from allennlp.common import Registrable

from topdown_parser.am_algebra import AMType
from topdown_parser.dataset_readers.AdditionalLexicon import AdditionalLexicon, Lexicon
from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.nn.EdgeLabelModel import EdgeLabelModel
from topdown_parser.nn.utils import get_device_id
from topdown_parser.transition_systems.parsing_state import BatchedParsingState


@dataclass(frozen=True)
class Decision:
    position : int
    pop: bool
    label : str
    supertag : Tuple[str, str]
    lexlabel : str
    termtyp : Optional[AMType] = None
    score: float = 0.0


@dataclass(frozen=True)
class DecisionBatch:
    push_tokens: torch.Tensor #shape (batch_size,)
    push_mask: torch.Tensor #shape (batch_size,) for which batch elements do we perform push?
    pop_mask: torch.Tensor #shape (batch_size,) shall we perform a pop operation?

    edge_labels: torch.Tensor #shape (batch_size,)

    constants: torch.Tensor #shape (batch_size,)
    term_types: Optional[torch.Tensor] #shape (batch_size,)
    lex_labels: torch.Tensor
    constant_mask: torch.Tensor

    @staticmethod
    def from_decision(decision : Decision, lexicon: AdditionalLexicon) -> "DecisionBatch":
        return DecisionBatch(torch.tensor([decision.position]),
                             torch.tensor([int(not decision.pop)]),
                             torch.tensor([int(decision.pop)]),
                             torch.tensor([lexicon.get_id("edge_labels", decision.label)]),
                             torch.tensor([lexicon.get_id("constants", "--TYPE--".join(decision.supertag))]),
                             torch.tensor([lexicon.get_id("term_types", str(decision.termtyp)) if decision.termtyp is not None else 0]),
                             torch.tensor([lexicon.get_id("lex_labels", decision.lexlabel)]),
                             torch.tensor([decision.supertag != ("", "")])
                             )

    def to(self, device):
        return DecisionBatch(self.push_tokens.to(device),
                             self.push_mask.to(device),
                             self.pop_mask.to(device),
                             self.edge_labels.to(device),
                             self.constants.to(device),
                             self.term_types.to(device),
                             self.lex_labels.to(device),
                             self.constant_mask.to(device))


class TransitionSystem(Registrable):

    def __init__(self, additional_lexicon : AdditionalLexicon):
        self.additional_lexicon = additional_lexicon

    def get_decision_batch_type(self) -> Type:
        return DecisionBatch

    def guarantees_well_typedness(self) -> bool:
        raise NotImplementedError()

    def validate_model(self, parser : "TopDownDependencyParser") -> None:
        """
        Check if the parsing model produces all the scores that we need.
        :param parser:
        :return:
        """
        return

    def get_order(self, sentence : AMSentence) -> Iterable[Decision]:
        """
        Pre-compute the sequence of decisions that parser should produce.
        The decisions use 1-based indexing for nodes.
        :param sentence:
        :return:
        """
        raise NotImplementedError()

    def check_correct(self, gold_sentence : AMSentence, predicted : AMSentence) -> bool:
        """
        Check if the predicted sentence is exactly the same as the gold sentence.
        Has to be implemented because not all transition system have to predict lexical as-graphs.
        :param gold_sentence:
        :param predicted:
        :return:
        """
        raise NotImplementedError()

    def initial_state(self, sentences : List[AMSentence], decoder_state : Any, device: Optional[int] = None) -> BatchedParsingState:
        raise NotImplementedError()

    def step(self, state: BatchedParsingState, decision_batch: DecisionBatch) -> None:
        """
        Applies a decision to a parsing state.
        :param state:
        :param decision_batch:
        :return:
        """
        raise NotImplementedError()

    def make_decision(self, scores: Dict[str, torch.Tensor], state : BatchedParsingState) -> DecisionBatch:
        """
        Informs the transition system about the last node chosen
        Returns the index of the node that will get a child next according to the transitions system.
        :param scores: additional choices for each batch element, like edge labels for example, contains edge existence scores.
        :return: a tensor of shape (batch_size,) of currently active nodes
            and a tensor of shape (batch_size, input_seq_len) which for every input position says if it is a valid next choice.
            input_seq_len is set above in reset_parses
        """
        raise NotImplementedError()

    def top_k_decision(self, scores: Dict[str, torch.Tensor], state: BatchedParsingState, k : int) -> DecisionBatch:
        raise NotImplementedError()

    def assumes_greedy_ok(self) -> Set[str]:
        """
        The dictionary keys of the context provider which we make greedy decisions on in top_k_decisions
        because we assume these choices won't impact future scores.
        TODO something's not right in terms of hierarchy: gather_context should be at level of ParsingState and not CommonParsingState?
        :return:
        """
        raise NotImplementedError()

    def predict_supertag_from_tos(self) -> bool:
        """
        shall we try to predict the supertag using the encoding of the nodes on top of the stack? (dfs-children-first)
        or shall we try to predict it from the nodes just selected? (dfs)
        :return:
        """
        return True
        #raise NotImplementedError()

    def decision_to_score(self, sentence : AMSentence, decision) -> Dict[str, torch.Tensor]:
        """
        In order to simulate scores for training data.
        :param sentence:
        :param decision:
        :return:
        """
        children_scores = torch.zeros(1, len(sentence)+1)
        children_scores[0, decision.position] = 1
        constant_scores = torch.zeros(1, self.additional_lexicon.vocab_size("constants"))
        if decision.supertag is not None:
            constant_scores[0, self.additional_lexicon.get_id("constants", "--TYPE--".join(decision.supertag))] = 1

        term_type_scores = torch.zeros(1, self.additional_lexicon.vocab_size("term_types"))
        if decision.termtyp is not None:
            term_type_scores[0, self.additional_lexicon.get_id("term_types", str(decision.termtyp))] = 1

        lex_label_scores = torch.zeros(1, self.additional_lexicon.vocab_size("lex_labels"))
        if decision.lexlabel != "":
            lex_label_scores[0, self.additional_lexicon.get_id("lex_labels", decision.lexlabel)] = 1

        edge_label_scores = torch.zeros(1, len(sentence)+1, self.additional_lexicon.vocab_size("edge_labels"))
        if decision.label != "":
            edge_label_scores[0, decision.position, self.additional_lexicon.get_id("edge_labels", decision.label)] = 1

        return {"children_scores": children_scores, "constants_scores": constant_scores,
                "term_types_scores": term_type_scores, "lex_labels_scores" : lex_label_scores,
                "lex_labels" : torch.tensor([self.additional_lexicon.get_id("lex_labels", decision.lexlabel)]),
                "all_labels_scores" : edge_label_scores}

    def fuzz_scores(self, sentence: AMSentence, beam_search: bool) -> Dict[str, torch.Tensor]:
        children_scores = torch.rand(1, len(sentence)+1)
        constant_scores = torch.rand(1, self.additional_lexicon.vocab_size("constants"))

        term_type_scores = torch.rand(1, self.additional_lexicon.vocab_size("term_types"))

        lex_label_scores = torch.rand(1, self.additional_lexicon.vocab_size("lex_labels"))

        r = {"children_scores": children_scores, "constants_scores": constant_scores,
             "lex_labels" : torch.randint(0, self.additional_lexicon.vocab_size("lex_labels"), (1,)),
                "term_types_scores": term_type_scores, "lex_labels_scores" : lex_label_scores}

        r["all_labels_scores"] = torch.rand((1, len(sentence)+1, self.additional_lexicon.vocab_size("edge_labels")))

        return r

