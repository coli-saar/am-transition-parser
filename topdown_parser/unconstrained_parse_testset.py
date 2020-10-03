import argparse
import json
import sys

import os
from allennlp.common.util import prepare_environment, import_submodules
from allennlp.models import load_archive
from allenpipeline import PipelineTrainerPieces
from allenpipeline.callback import CallbackName


# Example:
# python topdown_parser/parse_testset.py models/my_model --batch_size 32 --beams 1 2 3


if __name__ == "__main__":
    import_submodules("topdown_parser")
    from topdown_parser.dataset_readers.same_formalism_iterator import SameFormalismIterator
    from topdown_parser.callbacks.parse_test import ParseTest
    from topdown_parser.dataset_readers.amconll_tools import parse_amconll
    from topdown_parser.am_algebra.tools import is_welltyped
    from topdown_parser.transition_systems.ltl import LTL
    from topdown_parser.transition_systems.ltf import LTF
    from topdown_parser.transition_systems.dfs import DFS
    from topdown_parser.transition_systems.dfs_children_first import DFSChildrenFirst

    optparser = argparse.ArgumentParser(add_help=True,
                                        description="Parse an amconll file (no annotions) with beam search.")

    optparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
    optparser.add_argument('--cuda-device', type=int, default=0, help='id of GPU to use. Use -1 to compute on CPU.')
    optparser.add_argument('--beams', nargs="*", help='beam sizes to use.')
    optparser.add_argument("--batch_size", type=int, default=None, help="Overwrite batch size.")



    args = optparser.parse_args()

    if args.beams is None:
        args.beams = [1] #set to greedy only

    archive = load_archive(args.archive_file, args.cuda_device)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()
    pipelinepieces = PipelineTrainerPieces.from_params(config)

    if args.batch_size is not None and args.batch_size > 0:
        assert isinstance(pipelinepieces.annotator.data_iterator, SameFormalismIterator)
        iterator : SameFormalismIterator = pipelinepieces.annotator.data_iterator
        pipelinepieces.annotator.data_iterator = SameFormalismIterator(iterator.formalisms, args.batch_size)

    annotator = pipelinepieces.annotator

    parse_test : ParseTest = pipelinepieces.callbacks.callbacks[CallbackName.AFTER_TRAINING.value]
    parse_test.active = True
    metrics = dict()

    model.transition_system = model.transition_system.get_unconstrained_version()
    model_dir = os.path.dirname(args.archive_file)

    for beam_size in [int(s) for s in args.beams]:
        model.k_best = beam_size

        for i in range(len(parse_test.system_inputs)):
            filename = os.path.join(model_dir, f"unconstrained_test_{parse_test.names[i]}_k_{beam_size}.txt")
            annotator.annotate_file(model, parse_test.system_inputs[i], filename)
            cumulated_parse_time = 0.0
            well_typed = 0
            total = 0
            with open(filename) as f:
                for am_sentence in parse_amconll(f, False):
                    cumulated_parse_time += float(am_sentence.attributes["normalized_parsing_time"])
                    well_typed += int(is_welltyped(am_sentence))
                    total += 1

            results = parse_test.test_commands[i].evaluate(filename)
            metrics.update({"test_"+parse_test.names[i]+"_k_"+str(beam_size)+"_"+name : val for name, val in results.items()})
            metrics["time_"+parse_test.names[i]+"_k_"+str(beam_size)] = cumulated_parse_time
            metrics["well_typed_"+parse_test.names[i]+"_k_"+str(beam_size)+"_percent"] = (well_typed/total) * 100

    print("Metrics", metrics)
    with open(os.path.join(model_dir, "unconstrained_test_metrics.json"), "w") as f:
        f.write(json.dumps(metrics))


