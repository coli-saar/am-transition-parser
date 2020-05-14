from typing import List, Optional, Dict, Set, Tuple

import torch
import torch.nn.functional as F

from topdown_parser.dataset_readers.AdditionalLexicon import AdditionalLexicon
import numpy as np


def single_score_to_selection(additional_scores : Dict[str, torch.Tensor], lexicon : AdditionalLexicon, namespace : str) -> Optional[Tuple[float, str]]:
    if namespace+"_scores" not in additional_scores:
        return None
    val, id = torch.max(additional_scores[namespace+"_scores"], dim=0)
    return val.cpu().numpy(), lexicon.get_str_repr(namespace, int(id))

def scores_to_selection(additional_scores : Dict[str, torch.Tensor], lexicon : AdditionalLexicon, namespace : str) -> Optional[List[str]]:
    if namespace+"_scores" not in additional_scores:
        return None
    return [lexicon.get_str_repr(namespace, int(id)) for id in torch.argmax(additional_scores[namespace+"_scores"], 1).cpu().numpy()]


def get_and_convert_to_numpy(additional_scores : Dict[str, torch.Tensor], key : str) -> Optional[np.array]:
    if key in additional_scores:
        return additional_scores[key].cpu().numpy()
    return None

def get_best_constant(correct_types: Set[int], constant_scores : np.array) -> Tuple[int, float]:
    """
    Returns the best constant in the set of correct_types
    :param correct_types:
    :param constant_scores: shape (constant vocab size)
    :return:
    """
    correct_types_it = iter(correct_types)
    best_index = next(correct_types_it)
    best_score = constant_scores[best_index]
    for i in correct_types_it:
        score = constant_scores[i]
        if score > best_score:
            best_index = i
            best_score = score
    return best_index, best_score
