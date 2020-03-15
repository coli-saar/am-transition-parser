from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict, Any, Set

import torch

from topdown_parser.am_algebra.tree import Tree
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

    def __init__(self, children_order: str):
        """
        Select children_order : "LR" (left to right) or "IO" (inside-out, recommended by Ma et al.)
        """
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
        self.sentences = sentences

    def step(self, selected_nodes: torch.Tensor, selected_labels: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = get_device_id(selected_nodes)
        assert selected_nodes.shape == (self.batch_size,)

        selected_nodes = selected_nodes.cpu().numpy()

        r = []
        next_choices = []
        for i in range(self.batch_size):
            if self.stack[i]:
                selected_node_in_batch_element = int(selected_nodes[i])
                if selected_node_in_batch_element in self.seen[i]:
                    popped = self.stack[i].pop()
                    assert popped == selected_nodes[i]
                else:
                    self.heads[i][selected_node_in_batch_element - 1] = self.stack[i][-1]

                    self.children[i][self.stack[i][-1]].append(selected_node_in_batch_element)  # 1-based

                    self.labels[i][selected_node_in_batch_element - 1] = selected_labels[i]

                    # push onto stack
                    self.stack[i].append(selected_node_in_batch_element)

                self.seen[i].add(selected_node_in_batch_element)
                choices = torch.zeros(self.input_seq_len)

                if not self.stack[i]:
                    r.append(0)
                else:
                    r.append(self.stack[i][-1])
                    choices[self.stack[i][-1]] = 1  # we can close the current node

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

        self.sentences = None
        self.stack = None
        self.seen = None
        self.heads = None
        self.batch_size = None

        return sentences

    def gather_context(self, active_nodes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """

        :param active_nodes: tensor of shape (batch_size,) with nodes that are currently on top of the stack.
        :return:
        """
        device = get_device_id(active_nodes)

        assert active_nodes.shape == (self.batch_size,)
        active_nodes = active_nodes.detach().cpu().numpy()

        grandparents = []
        siblings = []
        for i in range(self.batch_size):
            current_node = int(active_nodes[i])
            grandparents.append(get_parent(self.heads[i], current_node))
            siblings.append(get_siblings(self.children[i], self.heads[i], current_node))

        max_no_siblings = max(len(n) for n in siblings)

        with torch.no_grad():
            ret = {"parents": torch.tensor(grandparents, device=device)}
            sibling_tensor = torch.zeros((self.batch_size, max(1,max_no_siblings)), dtype=torch.long, device=device)
            for i in range(self.batch_size):
                for j, sibling in enumerate(siblings[i]):
                    sibling_tensor[i, j] = sibling
            ret["siblings"] = sibling_tensor
            ret[
                "siblings_mask"] = sibling_tensor != 0  # we initialized with 0 and 0 cannot possibly be a sibling of a node, because its the artificial root.
            return ret

    def undo_one_batching(self, context: Dict[str, torch.Tensor]) -> None:
        # context["parents"] has size (batch_size, decision seq len, 1)
        context["parents"] = context["parents"].squeeze(2)
