from copy import deepcopy
from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict, Any, Set

import torch

from topdown_parser.am_algebra.tree import Tree
from topdown_parser.dataset_readers.AdditionalLexicon import AdditionalLexicon
from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.nn.EdgeLabelModel import EdgeLabelModel
from topdown_parser.nn.utils import get_device_id
from topdown_parser.transition_systems.parsing_state import CommonParsingState, ParsingState
from topdown_parser.transition_systems.transition_system import TransitionSystem, Decision
from topdown_parser.transition_systems.utils import scores_to_selection, single_score_to_selection

import numpy as np

def flatten(l: Iterable[List[Any]]) -> List[Any]:
    r = []
    for x in l:
        r.extend(x)
    return r

class DFSState(CommonParsingState):

    def __init__(self, sentence_id: int, decoder_state: Any, active_node: int, score: float,
                 sentence : AMSentence, lexicon : AdditionalLexicon,
                 heads: List[int], children: Dict[int, List[int]], edge_labels: List[str],
                 constants : List[Tuple[str,str]], lex_labels : List[str],
                 stack : List[int], seen : Set[int]):

        super().__init__(sentence_id, decoder_state, active_node, score, sentence, lexicon, heads, children, edge_labels,
                         constants, lex_labels)

        self.stack = stack
        self.seen = seen

    def is_complete(self) -> bool:
        return self.stack == []



@TransitionSystem.register("dfs")
class DFS(TransitionSystem):

    def __init__(self, children_order: str, pop_with_0: bool, additional_lexicon : AdditionalLexicon):
        """
        Select children_order : "LR" (left to right) or "IO" (inside-out, recommended by Ma et al.)
        """
        super().__init__(additional_lexicon)
        self.pop_with_0 = pop_with_0
        assert children_order in ["LR", "IO"], "unknown children order"

        self.children_order = children_order

    def predict_supertag_from_tos(self) -> bool:
        return False

    def _construct_seq(self, tree: Tree) -> List[Decision]:
        own_position = tree.node[0]
        to_left = []
        to_right = []
        for child in tree.children:
            if child.node[1].label == "IGNORE":
                continue

            if child.node[0] < own_position:
                to_left.append(self._construct_seq(child))
            else:
                to_right.append(self._construct_seq(child))

        beginning = [Decision(own_position, tree.node[1].label, (tree.node[1].fragment, tree.node[1].typ), tree.node[1].lexlabel )]

        if self.pop_with_0:
            ending = [Decision(0, "", ("",""), "")]
        else:
            ending = [Decision(own_position, "", ("",""), "")]

        if self.children_order == "LR":
            return beginning + flatten(to_left) + flatten(to_right) + ending
        elif self.children_order == "IO":
            return beginning + flatten(reversed(to_left)) + flatten(to_right) + ending
        else:
            raise ValueError("Unknown children order: " + self.children_order)

    def get_order(self, sentence: AMSentence) -> Iterable[Decision]:
        t = Tree.from_am_sentence(sentence)
        r = self._construct_seq(t)
        return r

    def check_correct(self, gold_sentence : AMSentence, predicted : AMSentence) -> bool:
        return all(x.head == y.head for x, y in zip(gold_sentence, predicted)) and \
               all(x.label == y.label for x, y in zip(gold_sentence, predicted)) and \
               all(x.fragment == y.fragment and x.typ == y.typ for x, y in zip(gold_sentence, predicted))

    def initial_state(self, sentence : AMSentence, decoder_state : Any) -> ParsingState:
        stack = [0]
        seen = set()
        heads = [0 for _ in range(len(sentence))]
        children = {i: [] for i in range(len(sentence) + 1)}
        labels = ["IGNORE" for _ in range(len(sentence))]
        lex_labels = ["_" for _ in range(len(sentence))]
        supertags = [("_","_") for _ in range(len(sentence))]
        #TODO sentence id
        return DFSState(0, decoder_state, 0, 0.0, sentence,self.additional_lexicon, heads, children, labels, supertags, lex_labels, stack, seen)

    def step(self, state : DFSState, decision: Decision, in_place: bool = False) -> ParsingState:
        if in_place:
            copy = state
        else:
            copy = deepcopy(state)

        if state.stack:
            if decision.position == 0 and self.pop_with_0:
                copy.stack.pop()
            elif not self.pop_with_0 and decision.position in copy.seen:
                popped = copy.stack.pop()
                assert popped == decision.position
            else:
                copy.heads[decision.position-1] = copy.stack[-1]

                copy.children[copy.stack[-1]].append(decision.position)  # 1-based

                copy.edge_labels[decision.position - 1] = decision.label

                if copy.constants is not None:
                    copy.constants[decision.position - 1] = decision.supertag

                if copy.lex_labels is not None:
                    copy.lex_labels[decision.position - 1] = decision.lexlabel

                # push onto stack
                copy.stack.append(decision.position)

            copy.seen.add(decision.position)

            if not copy.stack:
                copy.active_node = 0
            else:
                copy.active_node = copy.stack[-1]
        else:
            copy.active_node = 0

        return copy

    def make_decision(self, scores: Dict[str, torch.Tensor], label_model: EdgeLabelModel, state : DFSState) -> Decision:
        # Select node:
        child_scores = scores["children_scores"].detach().cpu() # shape (input_seq_len)
        #Cannot select nodes that we have visited already (except if not pop with 0 and currently active, then we can close).

        for seen in state.seen:
            if self.pop_with_0 and seen == 0:
                pass
            elif not self.pop_with_0 and seen == state.active_node:
                pass
            else:
                child_scores[seen] = -10e10

        score = 0.0
        s, selected_node = torch.max(child_scores, dim=0)
        score += s
        s, selected_label = single_score_to_selection(scores, self.additional_lexicon, "edge_labels")
        score += s
        s, selected_supertag = single_score_to_selection(scores, self.additional_lexicon, "constants")
        score += s
        s, selected_lex_label = single_score_to_selection(scores, self.additional_lexicon, "lex_labels")
        score += s

        return Decision(int(selected_node), selected_label, AMSentence.split_supertag(selected_supertag), selected_lex_label, score=score)


    def top_k_decision(self, scores: Dict[str, torch.Tensor], encoder_state : Dict[str,torch.Tensor],
                       label_model: EdgeLabelModel, state: DFSState, k : int) -> List[Decision]:
        # Select node:
        child_scores = scores["children_scores"].detach().cpu() # shape (input_seq_len)
        #Cannot select nodes that we have visited already (except if not pop with 0 and currently active, then we can close).

        for seen in state.seen:
            if self.pop_with_0 and seen == 0:
                pass
            elif not self.pop_with_0 and seen == state.active_node:
                pass
            else:
                child_scores[seen] = -10e10

        scores, children = torch.argsort(child_scores, descending=True)
        scores = scores[:k]
        children = children[:k]
        # Now have k best children
        label_model.set_input(encoder_state["encoded_input"], encoder_state["input_mask"])
        label_model.edge_label_scores(state.decoder_state) #TODO duplicate decoder state

    def assumes_greedy_ok(self) -> Set[str]:
        """
        The dictionary keys of the context provider which we make greedy decisions on in top_k_decisions
        because we assume these choices won't impact future scores.
        :return:
        """
        return {"children_labels", "lexical_types"}
