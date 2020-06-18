import argparse
import os

import sys
from itertools import chain, combinations
from typing import Dict, Set

sys.path.append(".")
from topdown_parser.dataset_readers.amconll_tools import parse_amconll, AMSentence
from topdown_parser.am_algebra import ReadCache, NonAMTypeException, AMType
from topdown_parser.am_algebra.tree import Tree
from topdown_parser.am_algebra.tools import get_term_types, is_welltyped

optparser = argparse.ArgumentParser(add_help=True,
                                    description="")

optparser.add_argument('f',type=str,
                       help='file with graph constants')


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

args = optparser.parse_args()

lexical_types : Set[AMType] = set()

with open(args.f) as f:
    for line in f:
        line = line.rstrip("\n")
        constant, lex_type = AMSentence.split_supertag(line)
        lexical_types.add(AMType.parse_str(lex_type))

print("Number of lexical types", len(lexical_types))

all_apply_sets : Set[frozenset] = set()
empty_type = AMType.parse_str("()")
for typ in lexical_types:
    full_apply_set = typ.get_apply_set(empty_type)
    if full_apply_set is not None:
        for subset in powerset(full_apply_set):
            all_apply_sets.add(frozenset(subset))

print("Number of apply sets", len(all_apply_sets))