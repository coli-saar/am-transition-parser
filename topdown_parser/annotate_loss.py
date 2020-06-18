import argparse
from typing import List, Dict, Any, Optional

import torch
from allennlp.common.util import prepare_environment, import_module_and_submodules
from allennlp.data import Instance
from allennlp.models import load_archive
from allenpipeline import Annotator, OrderedDatasetReader
from allenpipeline.Decoder import split_up
import allennlp.nn.util as util


if __name__ == "__main__":
    import_module_and_submodules("topdown_parser")
    from topdown_parser.nn.parser import TopDownDependencyParser

    optparser = argparse.ArgumentParser(add_help=True,
                                        description="Annotates loss into an annotated amconll file")

    optparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
    optparser.add_argument('input_file', type=str, help='path to or url of the input file')
    optparser.add_argument('output_file', type=str, help='path to output file')
    optparser.add_argument('--cuda-device', type=int, default=0, help='id of GPU to use. Use -1 to compute on CPU.')


    args = optparser.parse_args()

    archive = load_archive(args.archive_file, args.cuda_device)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    annotator = Annotator.from_params(config["annotator"])

    annotator.annotate_file(model, args.input_file, args.output_file,
                            annotation_function=lambda model, model_input: model.annotate_loss(**model_input))

