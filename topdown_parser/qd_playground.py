from typing import List, Tuple

import torch

from topdown_parser.dataset_readers.AdditionalLexicon import Lexicon, AdditionalLexicon
from topdown_parser.dataset_readers.amconll_tools import AMSentence, parse_amconll
from topdown_parser.nn.qd_parser import tensorize_app_edges

if __name__ == "__main__":
    with open("data/AMR/2015/train/tiny.amconll") as f:
        sents = list(parse_amconll(f))

    lex = AdditionalLexicon({"edge_labels" : "data/AMR/2015/train/edges.txt"})

    i = 15
    b = 2
    some_sents = sents[i:i+b]
    tensors, mask = tensorize_app_edges(some_sents, lex.sublexica["edge_labels"])
    print(tensors)
    print("Mask")
    print(mask)
    for s in some_sents:
        print(s)
        print()
