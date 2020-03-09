from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple

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
        raise NotImplementedError()

    def get_active_nodes(self, sentence : AMSentence) -> Iterable[int]:
        raise NotImplementedError()

    def get_dep_tree(self, parsing_seq : List[Decision], sentence : AMSentence) -> AMSentence:
        raise NotImplementedError()

    def reset_parses(self, sentences : List[AMSentence], input_seq_len: int) -> None:
        raise NotImplementedError()

    def step(self, selected_nodes : torch.Tensor, selected_labels : List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Informs the transition system about the last node chosen
        Returns the index of the node that will get a child next according to the transitions system.
        :param selected_labels: a list of length batch_size with the current edge label predictions.
        :param selected_node: (batch_size,)
        :return: a tensor of shape (batch_size,)
            and a tensor of shape (batch_size, input_seq_len) which for every input position says if it is a valid next choice.
        """
        raise NotImplementedError()

    def retrieve_parses(self) -> List[AMSentence]:
        """
        Called after several calls to step.
        :return:
        """
        raise NotImplementedError()


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
        stack = []
        seen = set()
        for decision in decisions:
            position = decision.position
            if position in seen:
                stack.pop()
            else:
                stack.append(position)
            seen.add(position)
            if stack:
                yield stack[-1]


    def get_dep_tree(self, parsing_seq : List[Decision], sentence : AMSentence) -> AMSentence:
        stack = [-1]
        visited = {-1}
        heads = [0] * len(sentence.get_heads())
        labels = ["IGNORE"] * len(sentence.get_heads())

        for action in parsing_seq[1:]:
            position = action.position-1
            if position in visited:
                stack.pop()
            else:
                heads[position] = stack[-1]+1 #1-based indexing!
                labels[position] = action.label
                stack.append(position)
            visited.add(position)

        sentence = sentence.set_heads(heads)
        sentence = sentence.set_labels(labels)
        return sentence.set_heads(heads)

    def reset_parses(self, sentences : List[AMSentence], input_seq_len : int) -> None:
        self.input_seq_len = input_seq_len
        self.batch_size = len(sentences)
        self.stack = [[0] for _ in range(self.batch_size)]
        self.seen = [{0} for _ in range(self.batch_size)]
        self.heads = [[ 0 for _ in range(len(sentence.words))] for sentence in sentences]
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

                if selected_nodes[i] in self.seen[i]:
                    popped = self.stack[i].pop()
                    assert popped == selected_nodes[i]
                else:
                    self.heads[i][selected_nodes[i]-1] = self.stack[i][-1]
                    self.labels[i][selected_nodes[i]-1] = selected_labels[i]
                    self.stack[i].append(selected_nodes[i])

                self.seen[i].add(selected_nodes[i])
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


