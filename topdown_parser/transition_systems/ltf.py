# cython: language_level=3

from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict, Any, Set

import torch
from allennlp.common.checks import ConfigurationError

from topdown_parser.am_algebra import AMType, NonAMTypeException, new_amtypes
from topdown_parser.am_algebra.new_amtypes import CandidateLexType, ModCache
from topdown_parser.am_algebra.tools import get_term_types
from topdown_parser.am_algebra.tree import Tree
from topdown_parser.dataset_readers.AdditionalLexicon import AdditionalLexicon
from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.nn.utils import get_device_id
from topdown_parser.transition_systems import utils
from topdown_parser.transition_systems.transition_system import TransitionSystem, Decision, get_parent, get_siblings
from topdown_parser.transition_systems.utils import scores_to_selection, get_best_constant

import numpy as np


def flatten(l: Iterable[List[Any]]) -> List[Any]:
    r = []
    for x in l:
        r.extend(x)
    return r


def typ2supertag(lexicon : AdditionalLexicon) -> Dict[AMType, Set[int]]:
    _typ2supertag : Dict[AMType, Set[int]] = dict() #which supertags have the given type?

    for supertag, i in lexicon.sublexica["constants"]:
        _, typ = AMSentence.split_supertag(supertag)
        try:
            typ = AMType.parse_str(typ)
            if typ not in _typ2supertag:
                _typ2supertag[typ] = set()

            _typ2supertag[typ].add(i)
        except NonAMTypeException:
            print("Skipping type", typ)
    return _typ2supertag

def typ2i(additional_lexicon : AdditionalLexicon) -> Dict[AMType, int]:
    _typ2i :  Dict[AMType, int] = dict()
    for typ, i in additional_lexicon.sublexica["term_types"]:
        try:
            _typ2i[AMType.parse_str(typ)] = i
        except NonAMTypeException:
            pass
    return _typ2i

def collect_sources(additional_lexicon : AdditionalLexicon) -> Set[str]:
    sources: Set[str] = set()
    for label, _ in additional_lexicon.sublexica["edge_labels"]:
        if "_" in label:
            sources.add(label.split("_")[1])
    return sources


@TransitionSystem.register("ltf")
class LTF(TransitionSystem):
    """
    Lexical type first strategy.
    """

    def __init__(self, children_order: str, pop_with_0: bool, additional_lexicon : AdditionalLexicon):
        """
        Select children_order : "LR" (left to right) or "IO" (inside-out, recommended by Ma et al.)
        """
        super().__init__(additional_lexicon)
        self.pop_with_0 = pop_with_0
        assert children_order in ["LR", "IO"], "unknown children order"

        self.children_order = children_order

        self.typ2supertag : Dict[AMType, Set[int]] = typ2supertag(self.additional_lexicon)#which supertags have the given type?

        self.typ2i :  Dict[AMType, int] = typ2i(self.additional_lexicon) # which type has what id?

        self.candidate_lex_types = new_amtypes.CandidateLexType({typ for typ in self.typ2i.keys()})

        #self.subtype_cache = SubtypeCache(self.typ2i.keys())
        self.mod_cache = ModCache(self.typ2i.keys())

        self.sources: Set[str] = collect_sources(self.additional_lexicon)


    def predict_supertag_from_tos(self) -> bool:
        return False

    def validate_model(self, parser : "TopDownDependencyParser") -> None:
        """
        Check if the parsing model produces all the scores that we need.
        :param parser:
        :return:
        """
        if parser.term_type_tagger is None:
            raise ConfigurationError("ltf transition system requires term tagger")

        if parser.lex_label_tagger is None:
            raise ConfigurationError("ltf transition system requires lex label tagger")

        if parser.supertagger is None:
            raise ConfigurationError("ltf transition system requires tagger for constants")

    def get_order(self, sentence: AMSentence) -> Iterable[Decision]:
        t = Tree.from_am_sentence(sentence)
        term_types = get_term_types(t, sentence)

        def _construct_seq(tree: Tree) -> List[Decision]:
            own_position = tree.node[0]
            to_left = []
            to_right = []
            for child in tree.children:
                if child.node[1].label == "IGNORE":
                    continue

                if child.node[0] < own_position:
                    to_left.append(_construct_seq(child))
                else:
                    to_right.append(_construct_seq(child))
            incoming_edge = tree.node[1].label
            #beginning = [Decision(own_position, incoming_edge, (tree.node[1].fragment, tree.node[1].typ),
            #                      tree.node[1].lexlabel, term_types[own_position-1] if incoming_edge.startswith("MOD_") else None)]
            beginning = [Decision(own_position, incoming_edge, (tree.node[1].fragment, tree.node[1].typ),
                                  tree.node[1].lexlabel, term_types[own_position-1])]

            if self.pop_with_0:
                ending = [Decision(0, "", ("",""), "")]
            else:
                ending = [Decision(own_position, "", ("",""), "")]

            if self.children_order == "LR":
                return beginning + flatten(to_left) + flatten(to_right) + ending
            elif self.children_order == "IO":
                return beginning + flatten(reversed(to_left)) + flatten(to_right) + ending
            else:
                raise ValueError("Unknown children order: " + self.children_order)

        return _construct_seq(t)

    def reset_parses(self, sentences: List[AMSentence], input_seq_len: int) -> None:
        self.input_seq_len = input_seq_len
        self.batch_size = len(sentences)
        self.stack = [[0] for _ in range(self.batch_size)]  # 1-based
        self.seen = [{0} for _ in range(self.batch_size)]
        self.heads = [[0 for _ in range(len(sentence.words))] for sentence in sentences]
        self.children = [{i: [] for i in range(len(sentence.words) + 1)} for sentence in sentences]  # 1-based
        self.labels = [["IGNORE" for _ in range(len(sentence.words))] for sentence in sentences]
        self.lex_labels = [["_" for _ in range(len(sentence.words))] for sentence in sentences]
        self.supertags = [["_--TYPE--_" for _ in range(len(sentence.words))] for sentence in sentences]
        self.lexical_types = [[AMType.parse_str("_") for _ in range(len(sentence.words))] for sentence in sentences]
        self.applysets_todo = [[None for _ in range(len(sentence.words))] for sentence in sentences]
        self.words_left = [len(sentence.words) for sentence in sentences]
        self.sentences = sentences
        self.root_determined = [False for _ in sentences]

    def _sources_to_be_filled(self, i : int) -> int:
        s = 0
        for applyset_todo in self.applysets_todo[i]:
            if applyset_todo is not None:
                s += len(applyset_todo)
        return s

    def get_additional_choices(self, decision : Decision) -> Dict[str, List[str]]:
        """
        Turn a decision into a dictionary of additional choices (beyond the node that is selected)
        :param decision:
        :return:
        """
        r = dict()
        if decision.label != "":
            r["selected_edge_labels"] = [decision.label]
        if decision.supertag != ("",""):
            r["selected_constants"] = ["--TYPE--".join(decision.supertag)]
        if decision.termtyp is not None:
            r["selected_term_types"] = [decision.termtyp]
        if decision.lexlabel != "":
            r["selected_lex_labels"] = [decision.lexlabel]

        return r

    def step(self, selected_nodes: torch.Tensor, additional_scores : Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = get_device_id(selected_nodes)
        assert selected_nodes.shape == (self.batch_size,)

        selected_nodes = selected_nodes.cpu().numpy()
        label_scores = utils.get_and_convert_to_numpy(additional_scores, "edge_labels_scores") #shape (batch_size, label_vocab)
        constant_scores = utils.get_and_convert_to_numpy(additional_scores, "constants_scores") # shape (batch_size, constant)
        term_type_scores = utils.get_and_convert_to_numpy(additional_scores, "term_types_scores") # shpae (batch_size, type vocab)


        selected_lex_labels = scores_to_selection(additional_scores, self.additional_lexicon, "lex_labels")
        unconstrained_best_labels = scores_to_selection(additional_scores, self.additional_lexicon, "edge_labels")
        #unconstrained_best_constants = scores_to_selection(additional_scores, self.additional_lexicon, "constants")
        unconstrained_term_types = scores_to_selection(additional_scores, self.additional_lexicon, "term_types")

        r = []
        next_choices = []
        for i in range(self.batch_size):
            if self.stack[i]:
                selected_node_in_batch_element = int(selected_nodes[i])

                if selected_node_in_batch_element == 0 and self.pop_with_0:
                    self.stack[i].pop()
                elif not self.pop_with_0 and selected_node_in_batch_element in self.seen[i]:
                    popped = self.stack[i].pop()
                    assert popped == selected_nodes[i]
                else:
                    tos = self.stack[i][-1]
                    self.heads[i][selected_node_in_batch_element - 1] = tos

                    self.children[i][tos].append(selected_node_in_batch_element)  # 1-based

                    words_left_before_selection = self.words_left[i] - self._sources_to_be_filled(i)
                    self.words_left[i] -= 1

                    if not self.root_determined[i]:
                        self.labels[i][selected_node_in_batch_element-1] = "ROOT"
                        self.root_determined[i] = True

                        max_score = float("-inf")
                        best_constant = None
                        best_lex_type = None
                        for lex_type in self.candidate_lex_types.get_candidates(AMType.parse_str("()"), words_left_before_selection-1):
                            local_best_constant, local_best_constant_score = get_best_constant(self.typ2supertag[lex_type], constant_scores[i])
                            if local_best_constant_score >= max_score:
                                max_score = local_best_constant_score
                                best_constant = local_best_constant
                                best_lex_type = lex_type

                        self.supertags[i][selected_node_in_batch_element-1] = self.additional_lexicon.get_str_repr("constants", best_constant)
                        self.applysets_todo[i][selected_node_in_batch_element-1] = best_lex_type.get_apply_set(AMType.parse_str("()"))
                        self.lexical_types[i][selected_node_in_batch_element-1] = best_lex_type

                    else:
                        # What if we wanted to apply? What's the best apply operation?
                        lex_type_of_tos = self.lexical_types[i][tos-1]
                        max_apply_score = -np.inf
                        best_apply_source = None
                        best_apply_term_type = None
                        best_apply_constant = None
                        best_apply_lex_type = None

                        for todo_source in self.applysets_todo[i][tos-1]:
                            req = lex_type_of_tos.get_request(todo_source)
                            i_req = self.typ2i[req]
                            source_score = label_scores[i, self.additional_lexicon.get_id("edge_labels", "APP_"+todo_source)]

                            for lex_type in self.candidate_lex_types.get_candidates(req, words_left_before_selection):
                                best_local_constant, best_local_constant_score = get_best_constant(self.typ2supertag[lex_type], constant_scores[i])
                                score = source_score + best_local_constant_score + term_type_scores[i, i_req]

                                if score >= max_apply_score:
                                    best_apply_source = todo_source
                                    best_apply_constant = best_local_constant
                                    best_apply_term_type = req
                                    best_apply_lex_type = lex_type
                                    max_apply_score = score

                        # What if we want to modify? Find best mod operation.
                        max_mod_score = -np.inf
                        best_modify_source = None
                        best_modify_constant = None
                        best_modify_term_type = None
                        best_modify_lex_type = None

                        if words_left_before_selection > 0 and unconstrained_term_types is not None: #and \
                                #(best_apply_source is None or unconstrained_best_labels[i].startswith("MOD")):
                            # MOD is only allowed if it leaves enough words.

                            for source, subtype in self.mod_cache.get_modifiers(lex_type_of_tos):
                                local_subtype_score = term_type_scores[i, self.typ2i[subtype]]
                                local_label_score = label_scores[i, self.additional_lexicon.get_id("edge_labels", "MOD_"+source)]
                                local_score = local_subtype_score + local_label_score

                                for lex_type in self.candidate_lex_types.get_candidates(subtype, words_left_before_selection-1):
                                    # we use words_left_before_selection - 1 because by adding the MOD edge, we have one word less (this one)
                                    # and we didn't fill a source in the parent node that would compensate for that

                                    best_local_constant, best_local_constant_score = get_best_constant(self.typ2supertag[lex_type], constant_scores[i])
                                    score = local_score + best_local_constant_score
                                    if score > max_mod_score:
                                        max_mod_score = score
                                        best_modify_source = source
                                        best_modify_constant = best_local_constant
                                        best_modify_term_type = subtype
                                        best_modify_lex_type = lex_type

                        #if best_modify_source is not None:
                        #    # to make a fair comparison to the APP score, don't consider the term type score.
                        #    max_mod_score = label_scores[i, self.additional_lexicon.get_id("edge_labels", "MOD_"+best_modify_source)] + constant_scores[i, best_modify_constant]

                        #Make a decision on APP vs MOD
                        if max_mod_score > max_apply_score:
                            # MOD
                            self.labels[i][selected_node_in_batch_element-1] = "MOD_" + best_modify_source
                            self.supertags[i][selected_node_in_batch_element-1] = self.additional_lexicon.get_str_repr("constants", best_modify_constant)
                            self.applysets_todo[i][selected_node_in_batch_element-1] = best_modify_lex_type.get_apply_set(best_modify_term_type)
                            self.lexical_types[i][selected_node_in_batch_element-1] = best_modify_lex_type
                        else:
                            # APP
                            self.labels[i][selected_node_in_batch_element-1] = "APP_" + best_apply_source
                            self.supertags[i][selected_node_in_batch_element-1] = self.additional_lexicon.get_str_repr("constants", best_apply_constant)
                            self.applysets_todo[i][selected_node_in_batch_element-1] = best_apply_lex_type.get_apply_set(best_apply_term_type)
                            self.applysets_todo[i][tos-1].remove(best_apply_source)
                            self.lexical_types[i][selected_node_in_batch_element-1] = best_apply_lex_type

                    self.lex_labels[i][selected_node_in_batch_element-1] = selected_lex_labels[i]

                    # push onto stack
                    self.stack[i].append(selected_node_in_batch_element)

                # determine possible next choices for tokens
                self.seen[i].add(selected_node_in_batch_element)
                choices = torch.zeros(self.input_seq_len)

                if not self.stack[i]:
                    r.append(0)
                    next_choices.append(torch.zeros(self.input_seq_len))
                else:
                    r.append(self.stack[i][-1])

                    if self.stack[i][-1] == 0: #we are at the top level
                        if self.pop_with_0:
                            choices[0] = 1
                        else:
                            choices[self.stack[i][-1]] = 1

                    else:
                        if len(self.applysets_todo[i][self.stack[i][-1]-1]) == 0: # We can close the current node
                            if self.pop_with_0:
                                choices[0] = 1
                            else:
                                choices[self.stack[i][-1]] = 1

                        if len(self.applysets_todo[i][self.stack[i][-1]-1]) > 0 or self.words_left[i] - self._sources_to_be_filled(i) > 0:
                            # if we have to fill sources here
                            # or we CAN add more MOD edges because words might be left over.

                            for child in range(1, len(self.sentences[i]) + 1):  # or we can choose a node that has not been used yet
                                if child not in self.seen[i]:
                                    choices[child] = 1

                    next_choices.append(choices)
            else:
                r.append(0)
                next_choices.append(torch.zeros(self.input_seq_len))


        return torch.tensor(r, device=device), torch.stack(next_choices, dim=0)

    def check_correct(self, gold_sentence : AMSentence, predicted : AMSentence) -> bool:
        return all(x.head == y.head for x, y in zip(gold_sentence, predicted)) and \
               all(x.label == y.label for x, y in zip(gold_sentence, predicted)) and \
                all(x.fragment == y.fragment and x.typ == y.typ for x, y in zip(gold_sentence, predicted))

    def retrieve_parses(self) -> List[AMSentence]:
        assert all(stack == [] for stack in self.stack)  # all parses complete

        sentences = [sentence.set_heads(self.heads[i]) for i, sentence in enumerate(self.sentences)]
        sentences = [sentence.set_labels(self.labels[i]) for i, sentence in enumerate(sentences)]
        sentences = [sentence.set_supertags(self.supertags[i]) for i, sentence in enumerate(sentences)]
        sentences = [sentence.set_lexlabels(self.lex_labels[i]) for i, sentence in enumerate(sentences)]

        self.sentences = None
        self.stack = None
        self.seen = None
        self.heads = None
        self.batch_size = None
        self.lex_labels = None
        self.supertags = None
        self.root_determined = False

        return sentences

    def gather_context(self, active_nodes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """

        :param active_nodes: tensor of shape (batch_size,) with nodes that are currently on top of the stack.
        :return:
        """
        return super()._gather_context(active_nodes, self.batch_size, self.heads, self.children, self.labels, self.supertags)
