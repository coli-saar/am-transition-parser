from typing import Tuple

import torch

from topdown_parser.dataset_readers.amconll_tools import AMSentence


class ParsingState:

    def __init__(self, sentence_id : int):
        self.sentence_id = sentence_id

    def get_score(self) -> float:
        """
        Return score of state
        :return:
        """
        raise NotImplementedError()

    def get_decoder_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return state of LSTM for this sentence
        :return:
        """
        raise NotImplementedError()

    def get_active_node(self) -> int:
        """
        Node which will get children next.
        :return:
        """
        raise NotImplementedError()

    def is_complete(self) -> bool:
        raise NotImplementedError()

    def extract_tree(self) -> AMSentence:
        raise NotImplementedError()

