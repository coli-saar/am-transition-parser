from abc import ABC
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Set, Optional

import torch

from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.dataset_readers.AdditionalLexicon import AdditionalLexicon

import numpy as np

@dataclass
class ParsingState:

    sentence_id : int
    decoder_state : Any
    active_node : int
    score : float
    lexicon : AdditionalLexicon

    def is_complete(self) -> bool:
        raise NotImplementedError()

    def extract_tree(self) -> AMSentence:
        raise NotImplementedError()

    def gather_context(self, device : int) -> Dict[str, torch.Tensor]:
        """
        Extracts features of the current context like siblings, grandparents etc.
        :param device: id of device which to put on the tensors.
        :return: a dictionary which can be used for storing
        various kinds of information, we can condition on.
        """
        raise NotImplementedError()


class CommonParsingState(ParsingState, ABC):

    def __init__(self, sentence_id: int, decoder_state: Any, active_node: int, score: float,
                 sentence : AMSentence, lexicon : AdditionalLexicon,
                 heads: List[int], children: Dict[int, List[int]],
                 edge_labels : List[str], constants: Optional[List[Tuple[str,str]]] = None, lex_labels: Optional[List[str]] = None):

        super().__init__(sentence_id, decoder_state, active_node, score, lexicon)

        self.sentence = sentence
        self.heads = heads
        self.edge_labels = edge_labels
        self.constants = constants
        self.children = children
        self.lex_labels = lex_labels

    def extract_tree(self) -> AMSentence:
        sentence = self.sentence.set_heads(self.heads)
        sentence = sentence.set_labels(self.edge_labels)
        if self.constants is not None:
            sentence = sentence.set_supertag_tuples(self.constants)
        if self.lex_labels is not None:
            sentence = sentence.set_lexlabels(self.lex_labels)
        return sentence

    def gather_context(self, device) -> Dict[str, torch.Tensor]:
        """
        Helper function to gather the context
        :return:
        """

        siblings: List[int] = get_siblings(self.children, self.heads, self.active_node)
        labels_of_other_children = [self.edge_labels[child-1] for child in self.children[self.active_node]]

        if self.constants is not None:
            if self.active_node == 0:
                supertag_of_current_node = "_"
            else:
                _, typ = self.constants[self.active_node-1]
                supertag_of_current_node = typ

        no_siblings = len(siblings)

        with torch.no_grad():
            ret = {"parents": torch.tensor(np.array([get_parent(self.heads, self.active_node)]), device=device)}
            sibling_tensor = torch.zeros(max(1,no_siblings), dtype=torch.long, device=device)
            for j, sibling in enumerate(siblings):
                sibling_tensor[j] = sibling
            ret["siblings"] = sibling_tensor
            ret["siblings_mask"] = sibling_tensor != 0  # we initialized with 0 and 0 cannot possibly be a sibling of a node, because it's the artificial root.

            children_tensor = torch.zeros(max(1, len(self.children[self.active_node])), dtype=torch.long, device=device)
            for j, child in enumerate(self.children[self.active_node]):
                children_tensor[j] = child
            ret["children"] = children_tensor
            ret["children_mask"] = (children_tensor != 0) # 0 cannot be a child of a node.

            if "edge_labels" in self.lexicon.sublexica:
                # edge labels of other children:
                label_tensor = torch.zeros(max(1, len(labels_of_other_children)), dtype=torch.long, device=device)
                for j, label in enumerate(labels_of_other_children):
                    label_tensor[j] = self.lexicon.get_id("edge_labels", label)
                ret["children_labels"] = label_tensor
                #mask is children_mask

            if "term_types" in self.lexicon.sublexica and self.constants is not None:
                ret["lexical_types"] = torch.tensor(np.array([self.lexicon.get_id("term_types", supertag_of_current_node)]), dtype=torch.long, device=device)

            return ret

def undo_one_batching(context : Dict[str, torch.Tensor]) -> None:
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







