from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict, Any, Set

import torch

from topdown_parser.am_algebra import AMType, new_amtypes
from topdown_parser.am_algebra.new_amtypes import ByApplySet, ModCache
from topdown_parser.am_algebra.tree import Tree
from topdown_parser.dataset_readers.AdditionalLexicon import AdditionalLexicon
from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.nn.utils import get_device_id
from topdown_parser.transition_systems.ltf import typ2supertag, typ2i, collect_sources
from topdown_parser.transition_systems.transition_system import TransitionSystem, Decision, get_parent, get_siblings
from topdown_parser.transition_systems.utils import scores_to_selection, get_and_convert_to_numpy, get_best_constant

import numpy as np

@TransitionSystem.register("ltl")
class LTL(TransitionSystem):
    """
    DFS where when a node is visited for the second time, all its children are visited once.
    Afterwards, the first child is visited for the second time. Then the second child etc.
    """

    def __init__(self, children_order: str, pop_with_0: bool,
                 additional_lexicon: AdditionalLexicon,
                 reverse_push_actions: bool = False):
        """
        Select children_order : "LR" (left to right) or "IO" (inside-out, recommended by Ma et al.)
        reverse_push_actions means that the order of push actions is the opposite order in which the children of
        the node are recursively visited.
        """
        super().__init__(additional_lexicon)
        self.pop_with_0 = pop_with_0
        self.reverse_push_actions = reverse_push_actions
        assert children_order in ["LR", "IO", "RL"], "unknown children order"

        self.children_order = children_order

        self.typ2supertag : Dict[AMType, Set[int]] = typ2supertag(self.additional_lexicon)#which supertags have the given type?
        self.supertag2typ : Dict[int, AMType] = dict()

        for typ, constants in self.typ2supertag.items():
            for constant in constants:
                self.supertag2typ[constant] = typ

        self.typ2i :  Dict[AMType, int] = typ2i(self.additional_lexicon) # which type has what id?

        self.candidate_lex_types = new_amtypes.CandidateLexType({typ for typ in self.typ2i.keys()})

        self.sources: Set[str] = collect_sources(self.additional_lexicon)

        modify_sources = {source for source in self.sources if self.additional_lexicon.contains("edge_labels", "MOD_"+source) }
        self.modify_ids = {self.additional_lexicon.get_id("edge_labels", "MOD_"+source) for source in modify_sources} #ids of modify edges
        self.mod_cache = ModCache(self.typ2i.keys())

        self.apply_cache = ByApplySet(self.typ2i.keys())

    def predict_supertag_from_tos(self) -> bool:
        return True

    def _construct_seq(self, tree: Tree) -> List[Decision]:
        own_position = tree.node[0]
        push_actions = []
        recursive_actions = []

        if self.children_order == "LR":
            children = tree.children
        elif self.children_order == "RL":
            children = reversed(tree.children)
        elif self.children_order == "IO":
            left_part = []
            right_part = []
            for child in tree.children:
                if child.node[0] < own_position:
                    left_part.append(child)
                else:
                    right_part.append(child)
            children = list(reversed(left_part)) + right_part
        else:
            raise ValueError("Unknown children order: "+self.children_order)

        for child in children:
            if child.node[1].label == "IGNORE":
                continue

            push_actions.append(Decision(child.node[0], child.node[1].label, ("", ""), ""))
            recursive_actions.extend(self._construct_seq(child))

        if self.pop_with_0:
            relevant_position = 0
        else:
            relevant_position = own_position

        if self.reverse_push_actions:
            push_actions = list(reversed(push_actions))

        return push_actions + [Decision(relevant_position, "", (tree.node[1].fragment, tree.node[1].typ), tree.node[1].lexlabel)] + recursive_actions

    def get_order(self, sentence: AMSentence) -> Iterable[Decision]:
        t = Tree.from_am_sentence(sentence)
        r = [Decision(t.node[0], t.node[1].label, ("", ""), "")] + self._construct_seq(t)
        return r

    def check_correct(self, gold_sentence : AMSentence, predicted : AMSentence) -> bool:
        return all(x.head == y.head for x, y in zip(gold_sentence, predicted)) and \
               all(x.label == y.label for x, y in zip(gold_sentence, predicted)) and \
               all(x.fragment == y.fragment and x.typ == y.typ for x, y in zip(gold_sentence, predicted))

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
        if decision.lexlabel != "":
            r["selected_lex_labels"] = [decision.lexlabel]

        return r

    def reset_parses(self, sentences: List[AMSentence], input_seq_len: int) -> None:
        self.input_seq_len = input_seq_len
        self.batch_size = len(sentences)
        self.stack = [[0] for _ in range(self.batch_size)]  # 1-based
        self.sub_stack = [[] for _ in range(self.batch_size)]  # 1-based
        self.seen = [{0} for _ in range(self.batch_size)]
        self.heads = [[0 for _ in range(len(sentence.words))] for sentence in sentences]
        self.children = [{i: [] for i in range(len(sentence.words) + 1)} for sentence in sentences]  # 1-based
        self.labels = [["IGNORE" for _ in range(len(sentence.words))] for sentence in sentences]
        self.lex_labels = [["_" for _ in range(len(sentence.words))] for sentence in sentences]
        self.supertags = [["_--TYPE--_" for _ in range(len(sentence.words))] for sentence in sentences]
        self.sentences = sentences
        self.lexical_types = [[AMType.parse_str("_") for _ in range(len(sentence.words))] for sentence in sentences]
        self.term_types = [[None for _ in range(len(sentence.words))] for sentence in sentences]
        self.applysets_collected = [[None for _ in range(len(sentence.words))] for sentence in sentences]
        self.sources_still_to_fill = [[0 for _ in range(len(sentence.words))] for sentence in sentences]
        self.words_left = [len(sentence.words) for sentence in sentences]
        self.sentences = sentences
        self.root_determined = [False for _ in sentences]
        self._step = [0 for _ in sentences]

    def step(self, selected_nodes: torch.Tensor, additional_scores : Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = get_device_id(selected_nodes)
        assert selected_nodes.shape == (self.batch_size,)

        selected_nodes = selected_nodes.cpu().numpy()
        label_scores = get_and_convert_to_numpy(additional_scores, "edge_labels_scores") #shape (batch_size, label_vocab)
        constant_scores = get_and_convert_to_numpy(additional_scores, "constants_scores") # shape (batch_size, constant)


        selected_lex_labels = scores_to_selection(additional_scores, self.additional_lexicon, "lex_labels")
        #unconstrained_best_labels = scores_to_selection(additional_scores, self.additional_lexicon, "edge_labels")
        #unconstrained_best_constants = scores_to_selection(additional_scores, self.additional_lexicon, "constants")

        r = []
        next_choices = []
        for i in range(self.batch_size):
            self._step[i] = self._step[i] + 1
            if self.stack[i]:
                selected_node_in_batch_element = int(selected_nodes[i])
                smallest_apply_set = 0

                if (selected_node_in_batch_element in self.seen[i] and not self.pop_with_0) or \
                        (selected_node_in_batch_element == 0 and self.pop_with_0) and len(self.stack[i]) == 1 and  self.stack[i][-1] == 0:
                    # Pop artificial root node
                    self.stack[i].pop()
                    if self.reverse_push_actions:
                            self.stack[i].extend(self.sub_stack[i])
                    else:
                        self.stack[i].extend(reversed(self.sub_stack[i]))
                    self.sub_stack[i] = []
                elif (selected_node_in_batch_element in self.seen[i] and not self.pop_with_0) or \
                    (selected_node_in_batch_element == 0 and self.pop_with_0):
                    tos = self.stack[i].pop()

                    if not self.pop_with_0:
                        assert tos == selected_nodes[i]

                    # determine lexical type:
                    max_score = -np.inf
                    best_constant = None
                    for term_typ in self.term_types[i][tos-1]:
                        possible_lex_types = self.apply_cache.by_apply_set(term_typ, frozenset(self.applysets_collected[i][tos-1]))
                        if possible_lex_types:
                            possible_constants = {constant for lex_type in possible_lex_types for constant in self.typ2supertag[lex_type]}
                            constant, score = get_best_constant(possible_constants, constant_scores[i])
                            if score > max_score:
                                max_score = score
                                best_constant = constant

                    chosen_lex_type = self.supertag2typ[best_constant]
                    self.lexical_types[i][tos-1] = chosen_lex_type
                    self.supertags[i][tos-1] = self.additional_lexicon.get_str_repr("constants", best_constant)

                    if selected_lex_labels is not None:
                        self.lex_labels[i][tos-1] = selected_lex_labels[i]

                    # now determine term types of children
                    for child_id in self.children[i][tos]: # 1-based children
                        child_id -= 1 # 0-based children
                        label = self.labels[i][child_id]

                        self.applysets_collected[i][child_id] = set()

                        if label.startswith("APP_"):
                            # get request at source
                            source = label.split("_")[1]
                            req = chosen_lex_type.get_request(source)
                            self.term_types[i][child_id] = {req}

                        elif label.startswith("MOD_"):
                            source = label.split("_")[1]
                            self.term_types[i][child_id] = set(self.mod_cache.get_modifiers_with_source(chosen_lex_type, source))

                        elif label == "ROOT":
                            pass
                        else:
                            raise ValueError("Somehow the invalid edge label "+label+" was produced")

                    # Pop
                    if self.reverse_push_actions:
                        self.stack[i].extend(self.sub_stack[i])
                    else:
                        self.stack[i].extend(reversed(self.sub_stack[i]))
                    self.sub_stack[i] = []
                else:
                    self.heads[i][selected_node_in_batch_element - 1] = self.stack[i][-1]
                    tos = self.stack[i][-1]

                    self.children[i][self.stack[i][-1]].append(selected_node_in_batch_element)  # 1-based
                    words_left = self.words_left[i]
                    self.words_left[i] -= 1

                    if not self.root_determined[i]:
                        self.labels[i][selected_node_in_batch_element-1] = "ROOT"
                        self.root_determined[i] = True
                        self.term_types[i][selected_node_in_batch_element-1] = {AMType.parse_str("()")}
                        self.applysets_collected[i][selected_node_in_batch_element-1] = set()
                        smallest_apply_set = 0 # EMPTY type is in the type lexicon, we could stop now.
                        self.sources_still_to_fill[i][selected_node_in_batch_element - 1] = 0
                    else:
                        # Check APPLY
                        max_apply_score = -np.inf
                        best_apply_source = None
                        #best_lex_type = None # for debugging purposes
                        smallest_apply_set = self.sources_still_to_fill[i][tos - 1]

                        apply_of_tos = self.applysets_collected[i][tos-1]
                        for term_typ in self.term_types[i][tos-1]:
                            for lexical_type, apply_set in self.candidate_lex_types.get_candidates_with_apply_set(term_typ, apply_of_tos, words_left + len(apply_of_tos)):
                                rest_of_apply_set = apply_set - apply_of_tos

                                for source in rest_of_apply_set:
                                    source_score = label_scores[i, self.additional_lexicon.get_id("edge_labels", "APP_"+source)]
                                    if source_score > max_apply_score and len(rest_of_apply_set) <= words_left:
                                        max_apply_score = source_score
                                        best_apply_source = source
                                        #best_lex_type = lexical_type

                        # Check MODIFY
                        max_modify_score = -np.inf
                        best_modify_edge_id = None
                        if words_left - smallest_apply_set > 0:
                            best_modify_edge_id, max_modify_score = get_best_constant(self.modify_ids, label_scores[i])

                        # Apply our choice
                        if max_modify_score > max_apply_score:
                            # MOD
                            self.labels[i][selected_node_in_batch_element - 1] = self.additional_lexicon.get_str_repr("edge_labels",  best_modify_edge_id)
                        else:
                            # APP
                            self.labels[i][selected_node_in_batch_element - 1] = "APP_" + best_apply_source
                            self.applysets_collected[i][tos-1].add(best_apply_source)

                            # recompute minimum number of sources to fill, needed if we cannot immediately close this node
                            # because we have apply set so far {op1,op2} but the only lexical type available is (op1,op2,op3,op4,op5)
                            smallest_apply_set = np.inf
                            for term_typ in self.term_types[i][tos-1]:
                                for lexical_type, apply_set in self.candidate_lex_types.get_candidates_with_apply_set(term_typ,
                                                self.applysets_collected[i][tos-1], words_left + len(self.applysets_collected[i][tos-1])):

                                    rest_of_apply_set = apply_set - self.applysets_collected[i][tos-1]
                                    smallest_apply_set = min(smallest_apply_set, len(rest_of_apply_set))

                            self.sources_still_to_fill[i][tos - 1] = smallest_apply_set

                    # push onto stack
                    self.sub_stack[i].append(selected_node_in_batch_element)

                self.seen[i].add(selected_node_in_batch_element)
                choices = torch.zeros(self.input_seq_len)

                if not self.stack[i]:
                    r.append(0)
                else:
                    r.append(self.stack[i][-1])

                    if smallest_apply_set == 0: # should be true most of the time.
                        # we can close the current node:
                        if self.pop_with_0:
                            choices[0] = 1
                        else:
                            choices[self.stack[i][-1]] = 1

                if self._step[i] == 1: # second step is always 0
                    choices[0] = 1
                else:
                    for child in range(1, len(self.sentences[i]) + 1):  # or we can choose a node that has not been used yet
                        if child not in self.seen[i]:
                            choices[child] = 1
                next_choices.append(choices)
            else:
                r.append(0)
                next_choices.append(torch.zeros(self.input_seq_len))

        return torch.tensor(r, device=device), torch.stack(next_choices, dim=0)

    def retrieve_parses(self) -> List[AMSentence]:
        assert all(stack == [] for stack in self.stack)  # all parses complete

        sentences = [sentence.set_heads(self.heads[i]) for i, sentence in enumerate(self.sentences)]
        sentences = [sentence.set_labels(self.labels[i]) for i, sentence in enumerate(sentences)]
        sentences = [sentence.set_supertags(self.supertags[i]) for i, sentence in enumerate(sentences)]
        sentences = [sentence.set_lexlabels(self.lex_labels[i]) for i, sentence in enumerate(sentences)]

        self.sentences = None
        self.stack = None
        self.sub_stack = None
        self.seen = None
        self.heads = None
        self.batch_size = None
        self.supertags = None
        self.lex_labels = None

        return sentences

    def gather_context(self, active_nodes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """

        :param active_nodes: tensor of shape (batch_size,) with nodes that are currently on top of the stack.
        :return:
        """
        return super()._gather_context(active_nodes, self.batch_size, self.heads, self.children, self.labels)
