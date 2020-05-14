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
#from topdown_parser.transition_systems.parsing_state import get_parent, get_siblings
from topdown_parser.transition_systems.utils import scores_to_selection, single_score_to_selection


class DFSChildrenFirstState(CommonParsingState):

    def __init__(self, sentence_id: int, decoder_state: Any, active_node: int, score: float,
                 sentence : AMSentence, lexicon : AdditionalLexicon,
                 heads: List[int], children: Dict[int, List[int]], edge_labels: List[str],
                 constants : List[Tuple[str,str]], lex_labels : List[str],
                 stack : List[int], seen : Set[int], substack : List[int]):

        super().__init__(sentence_id, decoder_state, active_node, score, sentence, lexicon, heads, children, edge_labels,
                         constants, lex_labels)

        self.stack = stack
        self.seen = seen
        self.substack = substack

    def is_complete(self) -> bool:
        return self.stack == []

@TransitionSystem.register("dfs-children-first")
class DFSChildrenFirst(TransitionSystem):
    """
    DFS where when a node is visited for the second time, all its children are visited once.
    Afterwards, the first child is visited for the second time. Then the second child etc.
    """

    def __init__(self, children_order: str, pop_with_0: bool,
                 additional_lexicon: AdditionalLexicon,
                 reverse_push_actions: bool = False):
        """
        Select children_order : "LR" (left to right) or "IO" (inside-out, recommended by Ma et al.)
        reverse_push_actions means that the order of push actions is the opposite order in which the children of
        the node are recursively visited.
        """
        super().__init__(additional_lexicon)
        self.pop_with_0 = pop_with_0
        self.reverse_push_actions = reverse_push_actions
        assert children_order in ["LR", "IO", "RL"], "unknown children order"

        self.children_order = children_order

    def predict_supertag_from_tos(self) -> bool:
        return True

    def _construct_seq(self, tree: Tree) -> List[Decision]:
        own_position = tree.node[0]
        push_actions = []
        recursive_actions = []

        if self.children_order == "LR":
            children = tree.children
        elif self.children_order == "RL":
            children = reversed(tree.children)
        elif self.children_order == "IO":
            left_part = []
            right_part = []
            for child in tree.children:
                if child.node[0] < own_position:
                    left_part.append(child)
                else:
                    right_part.append(child)
            children = list(reversed(left_part)) + right_part
        else:
            raise ValueError("Unknown children order: "+self.children_order)

        for child in children:
            if child.node[1].label == "IGNORE":
                continue

            push_actions.append(Decision(child.node[0], child.node[1].label, ("", ""), ""))
            recursive_actions.extend(self._construct_seq(child))

        if self.pop_with_0:
            relevant_position = 0
        else:
            relevant_position = own_position

        if self.reverse_push_actions:
            push_actions = list(reversed(push_actions))

        return push_actions + [Decision(relevant_position, "", (tree.node[1].fragment, tree.node[1].typ), tree.node[1].lexlabel)] + recursive_actions

    def get_order(self, sentence: AMSentence) -> Iterable[Decision]:
        t = Tree.from_am_sentence(sentence)
        r = [Decision(t.node[0], t.node[1].label, ("", ""), "")] + self._construct_seq(t)
        return r

    def check_correct(self, gold_sentence : AMSentence, predicted : AMSentence) -> bool:
        return all(x.head == y.head for x, y in zip(gold_sentence, predicted)) and \
               all(x.label == y.label for x, y in zip(gold_sentence, predicted)) and \
               all(x.fragment == y.fragment and x.typ == y.typ for x, y in zip(gold_sentence, predicted))


    def initial_state(self, sentence : AMSentence, decoder_state : Any) -> ParsingState:
        stack = [0]
        seen = set()
        substack = []
        heads = [0 for _ in range(len(sentence))]
        children = {i: [] for i in range(len(sentence) + 1)}
        labels = ["IGNORE" for _ in range(len(sentence))]
        lex_labels = ["_" for _ in range(len(sentence))]
        supertags = [("_","_") for _ in range(len(sentence))]
        #TODO sentence id
        return DFSChildrenFirstState(0, decoder_state, 0, 0.0, sentence,self.additional_lexicon, heads,
                                    children, labels, supertags, lex_labels, stack, seen, substack)


    def step(self, state : DFSChildrenFirstState, decision: Decision, in_place: bool = False) -> ParsingState:
        raise NotImplementedError("There's still a bug here.")
        if in_place:
            copy = state
        else:
            copy = deepcopy(state)

        if state.stack:
            position = decision.position

            if (position in copy.seen and not self.pop_with_0) or \
                (position == 0 and self.pop_with_0):
                popped = copy.stack.pop()

                if not self.pop_with_0:
                    assert popped == position

                if copy.constants is not None:
                    copy.constants[position - 1] = decision.supertag

                if copy.lex_labels is not None:
                    copy.lex_labels[position - 1] = decision.lexlabel

                if self.reverse_push_actions:
                    copy.stack.extend(copy.substack)
                else:
                    copy.stack.extend(reversed(copy.substack))
                copy.substack = []
            else:
                copy.heads[position - 1] = copy.stack[-1]

                copy.children[copy.stack[-1]].append(position)  # 1-based

                copy.edge_labels[position - 1] = decision.label

                # push onto stack
                copy.substack.append(position)

            copy.seen.add(position)

        else:
            copy.active_node = 0
        return copy

    def make_decision(self, scores: Dict[str, torch.Tensor], label_model: EdgeLabelModel, state : DFSChildrenFirstState) -> Decision:
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
