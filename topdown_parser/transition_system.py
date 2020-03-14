from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict, Any, Set

import torch
from allennlp.common import Registrable

from topdown_parser.am_algebra.tree import Tree
from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.nn.utils import get_device_id


@dataclass(frozen=True)
class Decision:
    position : int
    label : str


class TransitionSystem(Registrable):

    def get_order(self, sentence : AMSentence) -> Iterable[Decision]:
        """
        Pre-compute the sequence of decisions that parser should produce.
        The decisions use 1-based indexing for nodes.
        :param sentence:
        :return:
        """
        raise NotImplementedError()

    def get_active_nodes(self, sentence : AMSentence) -> Iterable[int]:
        """
        Pre-compute the sequence of "active" nodes, e.g. nodes that are on top of the stack.
        The active nodes use 1-based indexing (i.e. you can derive them from self.get_order())
        :param sentence:
        :return:
        """
        raise NotImplementedError()

    def reset_parses(self, sentences : List[AMSentence], input_seq_len: int) -> None:
        """
        Set the sentences that will be parsed next.
        :param sentences:
        :param input_seq_len: the length of the tensor valid_choices that step() produces
        :return:
        """
        raise NotImplementedError()

    def step(self, selected_nodes : torch.Tensor, selected_labels : List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Informs the transition system about the last node chosen
        Returns the index of the node that will get a child next according to the transitions system.
        :param selected_labels: a list of length batch_size with the current edge label predictions.
        :param selected_nodes: (batch_size,), 1-based indexing
        :return: a tensor of shape (batch_size,) of currently active nodes
            and a tensor of shape (batch_size, input_seq_len) which for every input position says if it is a valid next choice.
            input_seq_len is set above in reset_parses
        """
        raise NotImplementedError()

    def retrieve_parses(self) -> List[AMSentence]:
        """
        Called after several calls to step.
        :return:
        """
        raise NotImplementedError()

    def gather_context(self, active_nodes : torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extracts features of the current context like siblings, grandparents etc.
        :param active_nodes: tensor of shape (batch_size,)
        :return: a dictionary with values of shape (batch_size, +) where additional dimensions can be used for storing
        various kinds of information.
        """
        raise NotImplementedError()


def get_parent(heads : List[int], node : int) -> int:
    """
    Helper function to get the grandparent of a node.
    :param heads:
    :param node: 1-based
    :return: grandparent, 1-based
    """
    parent = heads[node-1]
    return parent

def get_siblings(children : Dict[int, List[int]], heads : List[int], node: int) -> List[int]:
    """

    :param children: 1-based
    :param heads:
    :param node: 1-based
    :return: siblings, 1-based.
    """
    parent = heads[node-1]
    all_children_of_parent = list(children[parent])
    if node in all_children_of_parent:
        all_children_of_parent.remove(node)
    return all_children_of_parent


@TransitionSystem.register("dfs")
class DFS(TransitionSystem):

    def get_order(self, sentence : AMSentence) -> Iterable[Decision]:
        t = Tree.from_am_sentence(sentence)

        def f(node, children):
            label = node[1].label
            r = []
            if label != "IGNORE":
                r.append(Decision(node[0], label))
            for child in children:
                r.extend(child)
            if label != "IGNORE":
                r.append(Decision(node[0], ""))
            return r

        return t.fold(f)

    def get_active_nodes(self, sentence : AMSentence) -> Iterable[int]:
        decisions = self.get_order(sentence)
        stack : List[int] = []
        seen : Set[int] = set()
        for decision in decisions:
            position = decision.position
            if position in seen:
                stack.pop()
            else:
                stack.append(position)
            seen.add(position)
            if stack:
                yield stack[-1]

    def reset_parses(self, sentences : List[AMSentence], input_seq_len : int) -> None:
        self.input_seq_len = input_seq_len
        self.batch_size = len(sentences)
        self.stack = [[0] for _ in range(self.batch_size)] #1-based
        self.seen = [{0} for _ in range(self.batch_size)]
        self.heads = [[ 0 for _ in range(len(sentence.words))] for sentence in sentences]
        self.children = [{ i : [] for i in range(len(sentence.words)+1)} for sentence in sentences] #1-based
        self.labels = [[ "IGNORE" for _ in range(len(sentence.words))] for sentence in sentences]
        self.sentences = sentences

    def step(self, selected_nodes : torch.Tensor, selected_labels : List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
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
                    self.heads[i][selected_node_in_batch_element-1] = self.stack[i][-1]

                    self.children[i][self.stack[i][-1]].append(selected_node_in_batch_element) #1-based

                    self.labels[i][selected_node_in_batch_element-1] = selected_labels[i]

                    # push onto stack
                    self.stack[i].append(selected_node_in_batch_element)

                self.seen[i].add(selected_node_in_batch_element)
                choices = torch.zeros(self.input_seq_len)

                if not self.stack[i]:
                    r.append(0)
                else:
                    r.append(self.stack[i][-1])
                    choices[self.stack[i][-1]] = 1 #we can close the current node

                for child in range(1,len(self.sentences[i])+1): #or we can choose a node that has not been used yet
                    if child not in self.seen[i]:
                        choices[child] = 1
                next_choices.append(choices)
            else:
                r.append(0)
                next_choices.append(torch.zeros(self.input_seq_len))

        return torch.tensor(r, device=device), torch.stack(next_choices, dim=0)

    def retrieve_parses(self) -> List[AMSentence]:
        assert all(stack==[] for stack in self.stack) #all parses complete

        sentences = [sentence.set_heads(self.heads[i]) for i,sentence in enumerate(self.sentences)]
        sentences = [sentence.set_labels(self.labels[i]) for i,sentence in enumerate(sentences)]

        self.sentences = None
        self.stack = None
        self.seen = None
        self.heads = None
        self.batch_size = None

        return sentences

    def gather_context(self, active_nodes : torch.Tensor) -> Dict[str, torch.Tensor]:
        """

        :param active_nodes: tensor of shape (batch_size,) with nodes that are currently on top of the stack.
        :return:
        """
        device = get_device_id(active_nodes)

        assert active_nodes.shape == (self.batch_size, )
        active_nodes = active_nodes.detach().cpu().numpy()

        grandparents = []
        siblings = []
        for i in range(self.batch_size):
            current_node = int(active_nodes[i])
            grandparents.append(get_parent(self.heads[i], current_node))
            siblings.append(get_siblings(self.children[i], self.heads[i], current_node))

        max_no_siblings = max(len(n) for n in siblings)

        with torch.no_grad():
            ret = {"parents" : torch.tensor(grandparents, device=device)}
            sibling_tensor = torch.zeros((self.batch_size, max_no_siblings))
            for i in range(self.batch_size):
                for j, sibling in enumerate(siblings[i]):
                    sibling_tensor[i,j] = sibling
            ret["siblings"] = sibling_tensor
            #ret["siblings_mask"] = sibling_tensor != 0 #we initialized with 0 and 0 cannot possibly be a sibling of a node, because its the artificial root.
            return ret




