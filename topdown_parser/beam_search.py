import argparse
import sys
import time
from typing import List, Dict, Any

import torch
from allennlp.common.util import prepare_environment, import_submodules
from allennlp.data import Instance
from allennlp.models import load_archive
from allenpipeline import Annotator, OrderedDatasetReader, PipelineTrainerPieces
from allenpipeline.Decoder import split_up
import allennlp.nn.util as util


if __name__ == "__main__":
    import_submodules("topdown_parser")

    optparser = argparse.ArgumentParser(add_help=True,
                                        description="Parse an amconll file (no annotions) with beam search.")

    optparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
    optparser.add_argument('input_file', type=str, help='path to or url of the input file')
    optparser.add_argument('output_file', type=str, help='path to output file')
    optparser.add_argument('--cuda-device', type=int, default=0, help='id of GPU to use. Use -1 to compute on CPU.')
    optparser.add_argument('--beam', type=int, default=2, help='beam size. Default: 2')

    args = optparser.parse_args()

    if args.beam < 1:
        print("Beam size must be at least 1")
        sys.exit()

    archive = load_archive(args.archive_file, args.cuda_device)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()
    model.k_best = args.beam

    pipelinepieces = PipelineTrainerPieces.from_params(config)

    t0 = time.time()
    pipelinepieces.annotator.annotate_file(model, args.input_file, args.output_file)
    t1 = time.time()
    print("Prediction took", t1-t0, "seconds")

