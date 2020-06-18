import argparse
import json
import sys

from allennlp.common.util import prepare_environment, import_module_and_submodules
from allennlp.models import load_archive
from allenpipeline import Annotator
from allenpipeline.callback import CallbackName, Callbacks, Callback

# Example:
# python topdown_parser/parse_testset.py models/my_model --batch_size 32 --beams 1 2 3


if __name__ == "__main__":
    import_module_and_submodules("topdown_parser")
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
    annotator = Annotator.from_params(config["annotator"])

    callbacks = Callbacks.from_params(config["trainer"]["external_callbacks"])
    parse_test : Callback= callbacks.callbacks[CallbackName.AFTER_TRAINING.value]
    parse_test : ParseTest
    parse_test.active = True
    metrics = dict()

    if isinstance(model.transition_system, LTL):
        print("Switching from LTL to DFSChildrenFirst")
        model.transition_system = DFSChildrenFirst(model.transition_system.children_order, model.transition_system.pop_with_0,
                                                   model.transition_system.additional_lexicon, model.transition_system.reverse_push_actions)
    elif isinstance(model.transition_system, LTF):
        print("Switching from LTF to DFS")
        model.transition_system = DFS(model.transition_system.children_order, model.transition_system.pop_with_0,
                                      model.transition_system.additional_lexicon)
    else:
        print("Model you specified was neither LTL nor LTF")
        sys.exit()

    for beam_size in [int(s) for s in args.beams]:
        model.k_best = beam_size

        for i in range(len(parse_test.system_inputs)):
            filename = args.archive_file+f"/test_{parse_test.names[i]}_k_{beam_size}.txt"
            annotator.annotate_file(model, parse_test.system_inputs[i], filename, batch_size=args.batch_size)
            cumulated_parse_time = 0.0
            well_typed = 0
            total = 0
            with open(filename) as f:
                for am_sentence in parse_amconll(f):
                    cumulated_parse_time += float(am_sentence.attributes["normalized_parsing_time"])
                    well_typed += int(is_welltyped(am_sentence))
                    total += 1

            results = parse_test.test_commands[i].evaluate(filename)
            metrics.update({"test_"+parse_test.names[i]+"_k_"+str(beam_size)+"_"+name : val for name, val in results.items()})
            metrics["time_"+parse_test.names[i]+"_k_"+str(beam_size)] = cumulated_parse_time
            metrics["well_typed_"+parse_test.names[i]+"_k_"+str(beam_size)+"_percent"] = (well_typed/total) * 100

    print("Metrics", metrics)
    with open(args.archive_file+"/unconstrained_test_metrics.json", "w") as f:
        f.write(json.dumps(metrics))


