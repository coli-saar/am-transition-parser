from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict, Any, Set

import torch

from topdown_parser.am_algebra.tree import Tree
from topdown_parser.dataset_readers.AdditionalLexicon import AdditionalLexicon
from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.nn.utils import get_device_id
from topdown_parser.transition_systems.transition_system import TransitionSystem, Decision, get_parent, get_siblings


def flatten(l: Iterable[List[Any]]) -> List[Any]:
    r = []
    for x in l:
        r.extend(x)
    return r


@TransitionSystem.register("dfs")
class DFS(TransitionSystem):

    def __init__(self, children_order: str, pop_with_0: bool, additional_lexicon : Optional[AdditionalLexicon] = None):
        """
        Select children_order : "LR" (left to right) or "IO" (inside-out, recommended by Ma et al.)
        """
        super().__init__(additional_lexicon)
        self.pop_with_0 = pop_with_0
        assert children_order in ["LR", "IO"], "unknown children order"

        self.children_order = children_order

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

    def reset_parses(self, sentences: List[AMSentence], input_seq_len: int) -> None:
        self.input_seq_len = input_seq_len
        self.batch_size = len(sentences)
        self.stack = [[0] for _ in range(self.batch_size)]  # 1-based
        self.seen = [{0} for _ in range(self.batch_size)]
        self.heads = [[0 for _ in range(len(sentence.words))] for sentence in sentences]
        self.children = [{i: [] for i in range(len(sentence.words) + 1)} for sentence in sentences]  # 1-based
        self.labels = [["IGNORE" for _ in range(len(sentence.words))] for sentence in sentences]
        self.lex_labels = [["_" for _ in range(len(sentence.words))] for sentence in sentences]
        self.supertags = [["_--TYPE--_" for _ in range(len(sentence.words))] for sentence in sentences]
        self.sentences = sentences

    def step(self, selected_nodes: torch.Tensor, additional_choices : Dict[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = get_device_id(selected_nodes)
        assert selected_nodes.shape == (self.batch_size,)

        selected_nodes = selected_nodes.cpu().numpy()
        selected_labels = additional_choices["selected_labels"]
        selected_supertags = additional_choices.get("selected_supertags", None)
        selected_lex_labels = additional_choices.get("selected_lex_labels", None)

        r = []
        next_choices = []
        for i in range(self.batch_size):
            if self.stack[i]:
                selected_node_in_batch_element = int(selected_nodes[i])

                if selected_node_in_batch_element == 0 and self.pop_with_0:
                    self.stack[i].pop()
                elif not self.pop_with_0 and selected_node_in_batch_element in self.seen[i]:
                    popped = self.stack[i].pop()
                    assert popped == selected_nodes[i]
                else:
                    self.heads[i][selected_node_in_batch_element - 1] = self.stack[i][-1]

                    self.children[i][self.stack[i][-1]].append(selected_node_in_batch_element)  # 1-based

                    self.labels[i][selected_node_in_batch_element - 1] = selected_labels[i]

                    if selected_supertags is not None:
                        self.supertags[i][selected_node_in_batch_element - 1] = selected_supertags[i]

                    if selected_lex_labels is not None:
                        self.lex_labels[i][selected_node_in_batch_element - 1] = selected_lex_labels[i]

                    # push onto stack
                    self.stack[i].append(selected_node_in_batch_element)

                self.seen[i].add(selected_node_in_batch_element)
                choices = torch.zeros(self.input_seq_len)

                if not self.stack[i]:
                    r.append(0)
                else:
                    r.append(self.stack[i][-1])
                    # We can close the current node:
                    if self.pop_with_0:
                        choices[0] = 1
                    else:
                        choices[self.stack[i][-1]] = 1

                for child in range(1, len(self.sentences[i]) + 1):  # or we can choose a node that has not been used yet
                    if child not in self.seen[i]:
                        choices[child] = 1
                next_choices.append(choices)
            else:
                r.append(0)
                next_choices.append(torch.zeros(self.input_seq_len))

        return torch.tensor(r, device=device), torch.stack(next_choices, dim=0)

    def retrieve_parses(self) -> List[AMSentence]:
        assert all(stack == [] for stack in self.stack)  # all parses complete

        sentences = [sentence.set_heads(self.heads[i]) for i, sentence in enumerate(self.sentences)]
        sentences = [sentence.set_labels(self.labels[i]) for i, sentence in enumerate(sentences)]
        sentences = [sentence.set_supertags(self.supertags[i]) for i, sentence in enumerate(sentences)]
        sentences = [sentence.set_lexlabels(self.lex_labels[i]) for i, sentence in enumerate(sentences)]

        self.sentences = None
        self.stack = None
        self.seen = None
        self.heads = None
        self.batch_size = None
        self.lex_labels = None
        self.supertags = None

        return sentences

    def gather_context(self, active_nodes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """

        :param active_nodes: tensor of shape (batch_size,) with nodes that are currently on top of the stack.
        :return:
        """
        return super()._gather_context(active_nodes, self.batch_size, self.heads, self.children, self.labels, self.supertags)
