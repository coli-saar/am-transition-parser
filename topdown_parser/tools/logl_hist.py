import argparse
import os

import sys
from typing import Dict, Set
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(".")
from topdown_parser.dataset_readers.amconll_tools import parse_amconll

optparser = argparse.ArgumentParser(add_help=True,
                                    description="reads two amconll files annotated with (negative) log likelihoods and shows histogram of difference.")
optparser.add_argument("system",help="System prediction")
optparser.add_argument("gold",help="Log likelihood on gold data.")
optparser.add_argument("bins",type=int, default=200, help="Log likelihood on gold data.")

args = optparser.parse_args()

logl_system = []
logl_gold = []

with open(args.system) as system_f:
    with open(args.gold) as gold_f:
        system_trees = parse_amconll(system_f)
        gold_trees = parse_amconll(gold_f)

        for st, gt in zip(system_trees, gold_trees):
            logl_system.append(float(st.attributes["loss"]))
            logl_gold.append(float(gt.attributes["loss"]))

logl_sytem = np.array(logl_system)
logl_gold = np.array(logl_gold)

diff = logl_gold - logl_system

print("Plotting gold - system")
print("e.g. a gold tree with loss 10 and system tree with loss 15 appears as -5.")
print("That is, the more negative numbers there are, the more beam search is needed.")
print("Number of trees where search error could be found (difference is negative)", np.sum(diff < 0))
axes = plt.axes()
axes.set_xlabel("Loss gold - system")
axes.set_ylabel("Count")
plt.hist(diff, bins=args.bins)
plt.show()