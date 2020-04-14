from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict, Any, Set

import torch
from allennlp.common import Registrable

from topdown_parser.dataset_readers.AdditionalLexicon import AdditionalLexicon
from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.nn.utils import get_device_id


@dataclass(frozen=True)
class Decision:
    position : int
    label : str
    supertag : Tuple[str, str]
    lexlabel : str


class TransitionSystem(Registrable):

    def __init__(self, additional_lexicon : Optional[AdditionalLexicon] = None):
        self.additional_lexicon = additional_lexicon

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

    def step(self, selected_nodes : torch.Tensor, additional_choices : Dict[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Informs the transition system about the last node chosen
        Returns the index of the node that will get a child next according to the transitions system.
        :param additional_choices: additional choices for each batch element, like edge labels for example.
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

    def _gather_context(self, active_nodes: torch.Tensor, batch_size : int,
                        heads : List[List[int]], children : List[Dict[int,List[int]]],
                        edge_labels : List[List[str]],
                        supertags : Optional[List[List[str]]] = None) -> Dict[str, torch.Tensor]:
        """
        Helper function to gather the context
        :param edge_labels: List[List[str]] edge labels for all nodes
        :param batch_size:
        :param heads: a list for the entire batch. Every element is a list of heads of the corresponding sentence
        :param children: a list for the entire batch. Every element is a dictionary mapping nodes to the list of their children. 1-based
        :param active_nodes: tensor of shape (batch_size,) with nodes that are currently on top of the stack.
        :return:
        """
        device = get_device_id(active_nodes)

        assert active_nodes.shape == (batch_size,)
        active_nodes = active_nodes.detach().cpu().numpy()

        grandparents : List[int] = []
        siblings : List[List[int]] = []
        other_children : List[List[int]] = []
        labels_of_other_children : List[List[str]] = []
        supertags_of_current_node : List[str] = []
        for i in range(batch_size):
            current_node = int(active_nodes[i])
            grandparents.append(get_parent(heads[i], current_node))
            siblings.append(get_siblings(children[i], heads[i], current_node))
            other_children.append(children[i][current_node])
            labels_of_other_children.append([edge_labels[i][child-1] for child in children[i][current_node]])

            if supertags is not None:
                if current_node == 0:
                    supertags_of_current_node.append("_")
                else:
                    _, typ = AMSentence.split_supertag(supertags[i][current_node-1])
                    supertags_of_current_node.append(typ)

        max_no_siblings = max(len(n) for n in siblings)

        with torch.no_grad():
            ret = {"parents": torch.tensor(grandparents, device=device)}
            sibling_tensor = torch.zeros((batch_size, max(1,max_no_siblings)), dtype=torch.long, device=device)
            for i in range(batch_size):
                for j, sibling in enumerate(siblings[i]):
                    sibling_tensor[i, j] = sibling
            ret["siblings"] = sibling_tensor
            ret["siblings_mask"] = sibling_tensor != 0  # we initialized with 0 and 0 cannot possibly be a sibling of a node, because its the artificial root.

            children_tensor = torch.zeros((batch_size, max(1, max( len(c) for c in other_children))), dtype=torch.long, device=device)
            for i in range(batch_size):
                for j, child in enumerate(other_children[i]):
                    children_tensor[i, j] = child
            ret["children"] = children_tensor
            ret["children_mask"] = (children_tensor != 0) # 0 cannot be a child of a node.

            if "edge_labels" in self.additional_lexicon.sublexica:
                # edge labels of other children:
                label_tensor = torch.zeros((batch_size, max(1, max(len(o) for o in labels_of_other_children))), dtype=torch.long, device=device)
                #shape (batch_size, max. number of children)
                for i in range(batch_size):
                    for j, label in enumerate(labels_of_other_children[i]):
                        label_tensor[i,j] = self.additional_lexicon.sublexica["edge_labels"].get_id(label)
                ret["children_labels"] = label_tensor
                #mask is children_mask

            if "lexical_types" in self.additional_lexicon.sublexica and supertags is not None:
                supertag_tensor = torch.tensor([self.additional_lexicon.sublexica["lexical_types"].get_id(t) for t in supertags_of_current_node], dtype=torch.long, device=device) #shape (batch_size,)
                ret["lexical_types"] = supertag_tensor

            return ret

    def undo_one_batching(self, context : Dict[str, torch.Tensor]) -> None:
        """
        Undo the effects introduced by gathering context with batch size 1 and batching them up.
        This will mostly mean: do nothing or remove dimensions with size 1.
        :param context:
        """
        # context["parents"] has size (batch_size, decision seq len, 1)
        context["parents"] = context["parents"].squeeze(2)

        if "lexical_types" in context:
            context["lexical_types"] = context["lexical_types"].squeeze(2)





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





