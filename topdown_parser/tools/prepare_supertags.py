import argparse
import os

import sys
sys.path.append("topdown_parser/dataset_readers/")
from amconll_tools import parse_amconll, AMSentence

optparser = argparse.ArgumentParser(add_help=True,
                                    description="reads amconll file and produces a file with the lexical types used")
optparser.add_argument("input", type=str)
optparser.add_argument("output", type=str, help="Where to write the types to")


args = optparser.parse_args()

supertags = set()

with open(args.input) as f:
    trees = parse_amconll(f)
    for tree in trees:
        supertags.update(AMSentence.split_supertag(s)[1] for s in tree.get_supertags())

with open(args.output,"w") as f:
    for t in sorted(supertags):
        f.write(t)
        f.write("\n")
