from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict, Any, Set

import torch
from allennlp.common import Registrable

from topdown_parser.dataset_readers.amconll_tools import AMSentence


@dataclass(frozen=True)
class Decision:
    position : int
    label : str
    supertag : Tuple[str, str]
    lexlabel : str


class TransitionSystem(Registrable):

    def get_order(self, sentence : AMSentence) -> Iterable[Decision]:
        """
        Pre-compute the sequence of decisions that parser should produce.
        The decisions use 1-based indexing for nodes.
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

    def undo_one_batching(self, context : Dict[str, torch.Tensor]) -> None:
        """
        Undo the effects introduced by gathering context with batch size 1 and batching them up.
        This will mostly mean: do nothing or remove dimensions with size 1.
        :param context:
        """
        return


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
    Helper function to get siblings of a node.
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





