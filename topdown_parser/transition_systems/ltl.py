import itertools
from dataclasses import dataclass
from typing import List, Iterable, Optional, Any, Dict, Set, Tuple, Type

import numpy as np
import torch
from tqdm import tqdm

from topdown_parser.am_algebra import AMType
from topdown_parser.am_algebra.new_amtypes import ModCache, combinations
from topdown_parser.am_algebra.tree import Tree
from topdown_parser.dataset_readers.AdditionalLexicon import AdditionalLexicon
from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.datastructures.list_of_list import BatchedListofList
from topdown_parser.datastructures.stack import BatchedStack
from topdown_parser.nn.utils import get_device_id
from topdown_parser.transition_systems.logic_torch import consistent_with_and_can_finish_now, tensor_or, index_OR, \
    batched_index_OR, debug_to_set
from topdown_parser.transition_systems.parsing_state import BatchedParsingState
from topdown_parser.transition_systems.transition_system import TransitionSystem, Decision, DecisionBatch


class LTLState(BatchedParsingState):

    def __init__(self,  decoder_state: Any,
                sentences: List[AMSentence],
                stack: BatchedStack,
                children: BatchedListofList,
                heads: torch.Tensor,
                edge_labels: torch.Tensor,
                constants: torch.Tensor,
                lex_labels: torch.Tensor,
                lexicon: AdditionalLexicon,
                lex_types : torch.Tensor,
                applyset : torch.Tensor
                 ):
        """

        :param decoder_state:
        :param sentences:
        :param stack:
        :param children:
        :param heads:
        :param edge_labels:
        :param constants:
        :param lex_labels:
        :param lexicon:
        :param lex_types: shape (batch_size, sent length)
        :param applyset: shape (batch_size, sent length, number of sources), belongs to currently active nodes
        """
        super(LTLState, self).__init__(decoder_state, sentences, stack, children, heads, edge_labels, constants, None, lex_labels, lexicon)
        self.lex_types = lex_types
        self.applyset = applyset
        self.step = 0

    def is_complete(self) -> torch.Tensor:
        return torch.all(self.stack.is_empty())

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
        self.pop_with_0=pop_with_0
        self.reverse_push_actions = reverse_push_actions
        assert children_order in ["LR", "IO", "RL"], "unknown children order"

        self.children_order = children_order
        self.i2source = sorted({label.split("_")[1] for label, _ in self.additional_lexicon.sublexica["edge_labels"] if "_" in label})
        self.source2i = {s: i for i, s in enumerate(self.i2source)}
        len_sources = len(self.i2source)

        self.additional_apps = ["APP_" + source for source in self.i2source if not self.additional_lexicon.contains("edge_labels", "APP_" + source)]
        self.additional_lexicon.sublexica["edge_labels"].add(self.additional_apps)
        len_labels = self.additional_lexicon.vocab_size("edge_labels")

        all_lex_types = {AMSentence.split_supertag(lextyp)[1] for lextyp, _ in self.additional_lexicon.sublexica["constants"] if "--TYPE--" in lextyp}
        self.i2lextyp = sorted(all_lex_types)
        self.lextyp2i : Dict[AMType, int] = { AMType.parse_str(l) : i for i, l in enumerate(self.i2lextyp)}
        len_lex_typ = len(self.i2lextyp)


        minimal_apply_sets = np.zeros((len_lex_typ, len_labels, len_lex_typ), dtype=np.long) #shape (parent lexical type, incoming label, lexical type)
        minimal_apply_sets += 100_000_000 #default value: cannot reach
        lexical2constant = np.zeros((len_lex_typ, self.additional_lexicon.vocab_size("constants")), dtype=np.bool) #shape (lexical type, constant)
        constant2lexical = np.zeros(self.additional_lexicon.vocab_size("constants"), dtype=np.long)
        apply_set_lookup = np.zeros((len_lex_typ, len_labels, len_lex_typ, len_sources), dtype=np.bool)
        apply_set_exists = np.zeros((len_lex_typ, len_labels, len_lex_typ), dtype=np.bool) #shape (parent lexical type, incoming label, lexical type)

        self.mod_cache = ModCache([AMType.parse_str(t) for t in all_lex_types])

        for constant,constant_id in self.additional_lexicon.sublexica["constants"]:
            lex_type = AMType.parse_str(AMSentence.split_supertag(constant)[1])
            lexical2constant[self.lextyp2i[lex_type], constant_id] = True
            constant2lexical[constant_id] = self.lextyp2i[lex_type]

        apply_reachable_from : Dict[AMType, Set[Tuple[AMType, frozenset]]] = dict()
        for t1 in self.lextyp2i.keys():
            if t1.is_bot:
                continue
            for t2 in self.lextyp2i.keys():
                if t2.is_bot:
                    continue
                applyset = t1.get_apply_set(t2)
                if applyset is not None:
                    if t2 not in apply_reachable_from:
                        apply_reachable_from[t2] = set()
                    apply_reachable_from[t2].add((t1, frozenset(applyset)))

        root_id = self.additional_lexicon.get_id("edge_labels", "ROOT")
        for parent_lex_typ, parent_id in tqdm(self.lextyp2i.items()):
            # ROOT
            # root requires empty term type, thus all sources must be removed
            for current_lex_type in self.lextyp2i.keys():
                if current_lex_type.is_bot:
                    continue
                current_typ_id = self.lextyp2i[current_lex_type]
                applyset_size = 0
                for source in current_lex_type.nodes():
                    applyset_size += 1
                    source_id = self.source2i[source]
                    apply_set_lookup[parent_id, root_id, current_typ_id, source_id] = True

                minimal_apply_sets[parent_id, root_id, current_typ_id] = applyset_size
                apply_set_exists[parent_id, root_id, current_typ_id] = True

            # MOD
            for source, t in self.mod_cache.get_modifiers(parent_lex_typ):
                if self.additional_lexicon.contains("edge_labels", "MOD_"+source):
                    label_id = self.additional_lexicon.get_id("edge_labels", "MOD_"+source)
                    for possible_lexical_type, applyset in apply_reachable_from[t]:
                        current_typ_id = self.lextyp2i[possible_lexical_type]

                        apply_set_exists[parent_id, label_id, current_typ_id] = True
                        for source in applyset:
                            source_id = self.source2i[source]
                            apply_set_lookup[parent_id, label_id, current_typ_id, source_id] = True
                        minimal_apply_sets[parent_id, label_id, current_typ_id] = min(len(applyset), minimal_apply_sets[parent_id, label_id, current_typ_id])
            # APP
            for source in parent_lex_typ.nodes():
                req = parent_lex_typ.get_request(source)
                label_id = self.additional_lexicon.get_id("edge_labels", "APP_"+source)
                for possible_lexical_type, applyset in apply_reachable_from[req]:
                    current_typ_id = self.lextyp2i[possible_lexical_type]

                    apply_set_exists[parent_id, label_id, current_typ_id] = True
                    for source in applyset:
                        source_id = self.source2i[source]
                        apply_set_lookup[parent_id, label_id, current_typ_id, source_id] = True
                    minimal_apply_sets[parent_id, label_id, current_typ_id] = min(len(applyset), minimal_apply_sets[parent_id, label_id, current_typ_id])

        self.minimal_apply_sets = torch.from_numpy(minimal_apply_sets)
        self.lexical2constant = torch.from_numpy(lexical2constant).int()
        self.constant2lexical = torch.from_numpy(constant2lexical)
        self.apply_set_lookup = torch.from_numpy(apply_set_lookup) #TODO GPU, float
        self.apply_set_exists = torch.from_numpy(apply_set_exists)
        self.app_source2label_id = torch.zeros((len_sources, len_labels), dtype=torch.bool) # maps a source id to the respective (APP) label id
        self.mod_tensor = torch.zeros(len_labels, dtype=torch.bool) #which label ids are MOD_ edge labels?
        self.label_id2appsource = torch.zeros(len_labels, dtype=torch.long)-1

        for label, label_id in self.additional_lexicon.sublexica["edge_labels"]:
            if label.startswith("MOD_"):
                self.mod_tensor[label_id] = True

        for source, source_id in self.source2i.items():
            label_id = self.additional_lexicon.get_id("edge_labels", "APP_"+source)
            self.label_id2appsource[label_id] = source_id
            self.app_source2label_id[source_id, label_id] = True

    def guarantees_well_typedness(self) -> bool:
        return False

    def add_missing_edge_scores(self, edge_scores : torch.Tensor) -> torch.Tensor:
        """
        Add edge scores for APP_x where x is a known source but APP_x was never seen in training
        :param edge_scores: shape (batch_size, number of edge_labels)
        :return:
        """
        additional_scores = torch.zeros((edge_scores.shape[0], len(self.additional_apps)), device=get_device_id(edge_scores))
        additional_scores -= 10_000_000 # unseen edge are very unlikely
        return torch.cat((edge_scores, additional_scores), dim=1)

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

            push_actions.append(Decision(child.node[0], False, child.node[1].label, ("", ""), ""))
            recursive_actions.extend(self._construct_seq(child))

        if self.pop_with_0:
            relevant_position = 0
        else:
            relevant_position = own_position

        if self.reverse_push_actions:
            push_actions = list(reversed(push_actions))

        return push_actions + [Decision(relevant_position, True, "", (tree.node[1].fragment, tree.node[1].typ), tree.node[1].lexlabel)] + recursive_actions

    def get_order(self, sentence: AMSentence) -> Iterable[Decision]:
        t = Tree.from_am_sentence(sentence)
        r = [Decision(t.node[0], False, t.node[1].label, ("", ""), "")] + self._construct_seq(t)
        return r

    def check_correct(self, gold_sentence : AMSentence, predicted : AMSentence) -> bool:
        return all(x.head == y.head for x, y in zip(gold_sentence, predicted)) and \
               all(x.label == y.label for x, y in zip(gold_sentence, predicted)) and \
               all(x.fragment == y.fragment and x.typ == y.typ for x, y in zip(gold_sentence, predicted))


    def initial_state(self, sentences : List[AMSentence], decoder_state : Any, device: Optional[int] = None) -> LTLState:
        max_len = max(len(s) for s in sentences)+1
        batch_size = len(sentences)
        stack = BatchedStack(batch_size, max_len+2, device=device)
        stack.push(torch.zeros(batch_size, dtype=torch.long, device=device), torch.ones(batch_size, dtype=torch.long, device=device))
        return LTLState(decoder_state, sentences, stack,
                                     BatchedListofList(batch_size, max_len, max_len, device=device),
                                     torch.zeros(batch_size, max_len, device=device, dtype=torch.long)-1, #heads
                                     torch.zeros(batch_size, max_len, device=device, dtype=torch.long), #labels
                                     torch.zeros(batch_size, max_len, device=device, dtype=torch.long)-1, #constants
                                     torch.zeros(batch_size, max_len, device=device, dtype=torch.long), #lex labels
                                     self.additional_lexicon,
                                     torch.zeros(batch_size, max_len, device=device, dtype=torch.long), #lexical types
                                     torch.zeros((batch_size, max_len, len(self.i2source)), device=device, dtype=torch.bool), #apply set
                                     )
        # decoder_state: Any
        # sentences : List[AMSentence]
        # stack: BatchedStack
        # children : BatchedListofList #shape (batch_size, input_seq_len, input_seq_len)
        # heads: torch.Tensor #shape (batch_size, input_seq_len) with 1-based id of parents, TO BE INITIALIZED WITH -1
        # edge_labels : torch.Tensor #shape (batch_size, input_seq_len) with the id of the incoming edge for each token
        # constants : torch.Tensor #shape (batch_size, input_seq_len)
        # lex_labels : torch.Tensor #shape (batch_size, input_seq_len)
        # lexicon : AdditionalLexicon

    def make_decision(self, scores: Dict[str, torch.Tensor], state : LTLState) -> DecisionBatch:
        children_scores = scores["children_scores"] #shape (batch_size, input_seq_len)
        batch_size, input_seq_len = children_scores.shape
        parent_mask = state.parent_mask()
        mask = parent_mask #shape (batch_size, input_seq_len)
        depth = state.stack.depth() #shape (batch_size,)
        active_nodes = state.stack.peek()
        batch_range = state.stack.batch_range
        applyset = state.applyset[batch_range, active_nodes]

        if state.step < 2:
            if state.step == 0: # nothing done yet, have to determine root
                # can only select a proper node when root not determined yet
                mask[:, 0] = 0
                push_mask = torch.ones(batch_size, dtype=torch.bool)
                edge_labels = self.additional_lexicon.get_id("edge_labels","ROOT") + torch.zeros(batch_size, device=get_device_id(children_scores), dtype=torch.long)
            elif state.step == 1:
                # second step is always selecting 0 (pop artificial root)
                mask = torch.zeros_like(mask)
                mask[:, 0] = 1
                push_mask = torch.zeros(batch_size, dtype=torch.bool)
                edge_labels = torch.argmax(scores["all_labels_scores"][:, 0], 1)

            mask = (1-mask.long())*10_000_000
            _, selected_nodes = torch.max(children_scores - mask, dim=1)

            constants = torch.zeros_like(edge_labels)
            lex_labels = scores["lex_labels"]

            return DecisionBatch(selected_nodes, push_mask, ~push_mask, edge_labels, constants, None, lex_labels, ~push_mask)

        parents = state.heads[batch_range, active_nodes] #shape (batch_size,)
        lexical_type_parent = state.lex_types[batch_range, parents] #shape (batch_size, )
        assert lexical_type_parent.shape == (batch_size,)
        incoming_labels = state.edge_labels[batch_range, active_nodes] #shape (batch_size,) with ids of incoming edge labels of active nodes

        A = self.apply_set_lookup[lexical_type_parent, incoming_labels]
        # A takes into account the set of term types of the currently active nodes
        # and tells us for each lexical type and source whether that source is in the apply set from the lexical type
        # to one of the term types that the current node can have.
        assert A.shape == (batch_size, len(self.lextyp2i), len(self.i2source))
        At = A.transpose(1, 2) #shape (batch_size, sources, lexical types)

        apply_set_exists = self.apply_set_exists[lexical_type_parent, incoming_labels] #shape (batch_size, lexical types, )
        minimal_apply_set_size = self.minimal_apply_sets[lexical_type_parent, incoming_labels] #shape (batch_size, lexical type,) with apply set size

        can_finish_now, consistent_lex_types = consistent_with_and_can_finish_now(applyset.int(), At.int(), apply_set_exists, minimal_apply_set_size) #both have shape (batch_size, lexical types) and are bool tensors
        assert can_finish_now.shape == (batch_size, len(self.lextyp2i))

        # TEST, there always has to be at least one lexical type that is consistent with what we have done.
        assert torch.sum(consistent_lex_types, dim=1) >= 1
        # TEST

        # we can potentially close the current node
        # if a) the stack is not empty already
        # and b) there is lexical type for our apply set and set of term types
        finishable = tensor_or(can_finish_now, dim=1) #shape (batch_size,)
        assert finishable.shape == (batch_size, )
        if self.pop_with_0:
            mask[batch_range, 0] *= (depth > 0)
            mask[:, 0] *= finishable
        else:
            mask[batch_range, active_nodes] *= (depth > 0)
            mask[batch_range, active_nodes] *= finishable


        mask = mask.long()
        mask *= state.position_mask()  # shape (batch_size, input_seq_len)

        mask = (1-mask)*10_000_000
        vals, selected_nodes = torch.max(children_scores - mask, dim=1)
        allowed_selection = vals > -1_000_000  # we selected something that was not extremely negative, shape (batch_size,)
        if self.pop_with_0:
            pop_mask = torch.eq(selected_nodes, 0)  #shape (batch_size,)
        else:
            pop_mask = torch.eq(selected_nodes, active_nodes)

        push_mask: torch.Tensor = (~pop_mask) * allowed_selection  # we push when we don't pop (but only if we are allowed to push)
        not_done = ~state.stack.get_done()
        push_mask *= not_done  # we can only push if we are not done with the sentence yet.
        pop_mask *= allowed_selection
        pop_mask *= not_done

        # compute constants for all instances (will only be used if pop_mask = True)
        # RE-USE the lexical types from above.
        possible_constants = index_OR(can_finish_now.int(), self.lexical2constant)
        assert possible_constants.shape == (batch_size, self.additional_lexicon.vocab_size("constants"))
        constant_mask = (1-possible_constants.float())*10_000_000
        selected_constants = torch.argmax(scores["constants_scores"]-constant_mask, dim=1) #shape (batch_size,)

        # Edge labels

        # We create masks for what edges are appropriate
        # MOD: all MOD_x edges are allowed provided that W_c - O_c >= 1

        # TODO get number of open sources, used for MOD = apply set collected - smallest apply set consistent with what we have collected
        # total_apply_set_sizes = LOOKUP[lexical_types_parent, incoming_labels] # (batch_size, candidate lex type) -> minimum over dim 1
        collected_apply_set_size = applyset.sum(dim=1) #shape (batch_size,)

        assert minimal_apply_set_size.shape == (batch_size, len(self.i2lextyp))
        non_consistent_lex_types = (1-consistent_lex_types.long())*10_000_000 #shape (batch_size, lexical type), at least one element is not extremely large, see assert above
        # non-consistent lexical types have huge number, all consistent types have 0 associated with them.

        # Find minimal apply set size of taking into account the lexical types that are consistent with out collected apply set.
        number_of_words_required = minimal_apply_set_size + non_consistent_lex_types #shape (batch_size, lexical type)
        minimal_apply_set_size, _ = torch.min(number_of_words_required, dim=1) #shape (batch_size,)

        #  we can use the fact that when the set of term types has the smallest apply set n and the largest apply set m, for all n <= i <= m, there is an apply set of size i.
        o_c = torch.relu(minimal_apply_set_size - collected_apply_set_size)
        w_c = state.lengths - (~parent_mask & state.position_mask().bool()).sum(dim=1) #shape (batch_size,)
        mod_mask = w_c - o_c >= 1 #shape (batch_size,)

        consistent_with_remaining_words = (number_of_words_required - collected_apply_set_size.unsqueeze(1)) <= w_c.unsqueeze(1) #shape (batch_size, lexical type)
        assert consistent_with_remaining_words.shape == (batch_size, len(self.i2lextyp))

        # TODO APP: all those APP_x such that, if we add x to our apply set
        # DONE: find lexical types that are CONSISTENT with apply set so far from type point of view (from above) --> (batch_size, lexical type)
        #  (parent lexical type, incoming label, lexical type) = size of smallest apply set --> (batch_size, lexical type) with size of smallest apply set
        #  find those lexical types where apply_set_covered - smallest_apply_set <= words left
        #  --> (batch_size, lexical type)
        #  (parent lexical type, incoming label, lexical type, source) = True iff source in applyset from lexical type to SOME term type
        #  --> (batch_size, source), use index_OR
        possible_app_sources = batched_index_OR(consistent_with_remaining_words.int(), A.int()) #shape (batch_size, sources)

        #  mask out all sources that have been used already
        possible_app_sources &= ~applyset

        #  translate source mask to edge label mask
        edge_mask = index_OR(possible_app_sources.int(), self.app_source2label_id.int()) #shape (batch_size, edge labels)

        edge_mask[mod_mask, :] |= self.mod_tensor #for some positions, we can also use MOD


        edge_scores = self.add_missing_edge_scores(scores["all_labels_scores"][state.stack.batch_range, selected_nodes]) #shape (batch_size, edge labels)

        edge_labels = torch.argmax(edge_scores - 10_000_000 * (1-edge_mask.float()), 1)

        lex_labels = scores["lex_labels"]

        return DecisionBatch(selected_nodes, push_mask, pop_mask, edge_labels, selected_constants, None, lex_labels, pop_mask)




    def step(self, state: LTLState, decision_batch: DecisionBatch) -> None:
        """
        Applies a decision to a parsing state.
        :param state:
        :param decision_batch:
        :return:
        """
        next_active_nodes = state.stack.peek()
        state.children.append(next_active_nodes, decision_batch.push_tokens, decision_batch.push_mask)
        range_batch_size = state.stack.batch_range
        inverse_push_mask = (1-decision_batch.push_mask.long())

        state.heads[range_batch_size, decision_batch.push_tokens] = inverse_push_mask*state.heads[range_batch_size, decision_batch.push_tokens] + decision_batch.push_mask * next_active_nodes
        state.edge_labels[range_batch_size, decision_batch.push_tokens] = inverse_push_mask*state.edge_labels[range_batch_size, decision_batch.push_tokens] + decision_batch.push_mask * decision_batch.edge_labels

        #Check if new edge labels are APP or MOD
        #if APP, add respective source to collected apply set.
        sources_used = self.label_id2appsource[decision_batch.edge_labels] #shape (batch_size,)
        app_mask = decision_batch.push_mask.bool() & (sources_used >= 0) #shape (batch_size,)
        state.applyset[range_batch_size, next_active_nodes, sources_used] = ~app_mask * state.applyset[range_batch_size, next_active_nodes, sources_used] + app_mask
        a = state.applyset.numpy()
        # apply_set_for_debugging = []
        # for batch in state.applyset.numpy():
        #     a = set()
        #     for i,x in enumerate(batch):
        #         if x:
        #             a.add(i)
        #     apply_set_for_debugging.append(a)

        lexical_types = self.constant2lexical[decision_batch.constants] #shape (batch_size,)
        inverse_constant_mask = (1-decision_batch.constant_mask.long())
        state.lex_types[range_batch_size, next_active_nodes] = inverse_constant_mask * state.lex_types[range_batch_size, next_active_nodes] + \
                                                                   decision_batch.constant_mask * lexical_types
        state.constants[range_batch_size, next_active_nodes] = inverse_constant_mask * state.constants[range_batch_size, next_active_nodes] + decision_batch.constant_mask * decision_batch.constants
        state.lex_labels[range_batch_size, next_active_nodes] = inverse_constant_mask*state.lex_labels[range_batch_size, next_active_nodes] + decision_batch.constant_mask * decision_batch.lex_labels

        pop_mask = decision_batch.pop_mask #shape (batch_size,)
        active_children = state.children.lol[range_batch_size, next_active_nodes] #shape (batch_size, max. number of children)
        push_all_children_mask = (active_children != 0).long() #shape (batch_size, max. number of children)
        push_all_children_mask *= pop_mask.unsqueeze(1) # only push those children where we will pop the current node from the top of the stack.

        state.stack.pop_and_push_multiple(active_children, decision_batch.pop_mask, push_all_children_mask, reverse=self.reverse_push_actions)

        state.step += 1



# Old code:
# def make_decision(self, scores: Dict[str, torch.Tensor], state : LTLState) -> Decision:
#     # Select node:
#     child_scores = scores["children_scores"].detach().cpu() # shape (input_seq_len)
#     INF = 10e10
#
#     if not state.root_determined: # First decision must choose root.
#         child_scores[0] = -INF
#         s, selected_node = torch.max(child_scores, dim=0)
#         return Decision(int(selected_node), "ROOT", ("",""), "",termtyp=None, score=float(s))
#
#     #Cannot select nodes that we have visited already.
#     for seen in state.seen:
#         if self.pop_with_0 and seen == 0:
#             pass
#         elif not self.pop_with_0 and seen == state.active_node:
#             pass
#         else:
#             child_scores[seen] = -INF
#
#     if state.active_node != 0 and state.sources_still_to_fill[state.active_node-1] > 0:
#         # Cannot close the current node if the smallest apply set reachable from the active node requires is still do add more APP edges.
#         if self.pop_with_0:
#             child_scores[0] = -INF
#         else:
#             child_scores[state.active_node] = -INF
#
#     score = 0.0
#     s, selected_node = torch.max(child_scores, dim=0)
#     score += s
#
#     if state.step == 1 or state.active_node == 0:
#         #we are done (or after first step), do nothing.
#         return Decision(0, "", ("",""), "", score=0.0)
#
#     if (selected_node in state.seen and not self.pop_with_0) or (selected_node == 0 and self.pop_with_0):
#         # pop node, select constant and lexical label.
#         constant_scores = scores["constants_scores"].cpu().numpy()
#         #max_score = -np.inf
#         #best_constant = None
#         possible_constants = set()
#         for term_typ in state.term_types[state.active_node-1]:
#             possible_lex_types = self.apply_cache.by_apply_set(term_typ, frozenset(state.applysets_collected[state.active_node-1]))
#             for lex_type in possible_lex_types:
#                 possible_constants.update(self.typ2supertag[lex_type])
#
#         assert len(possible_constants) > 0
#         best_constant, max_score = get_best_constant(possible_constants, constant_scores)
#         pop_node = 0 if self.pop_with_0 else state.active_node
#         selected_lex_label = self.additional_lexicon.get_str_repr("lex_labels", int(scores["lex_labels"].cpu().numpy()))
#         score += s
#         return Decision(pop_node, "", AMSentence.split_supertag(self.additional_lexicon.get_str_repr("constants", best_constant)), selected_lex_label, score=score)
#
#     # APP or MOD?
#     label_scores = scores["all_labels_scores"][selected_node].cpu().numpy() #shape (edge vocab size)
#
#     max_apply_score = -np.inf
#     #best_apply_source = None
#     #best_lex_type = None # for debugging purposes
#     smallest_apply_set = state.sources_still_to_fill[state.active_node - 1]
#
#     apply_of_tos = state.applysets_collected[state.active_node-1]
#     possible_sources = set()
#     for term_typ in state.term_types[state.active_node-1]:
#         for lexical_type, apply_set in self.candidate_lex_types.get_candidates_with_apply_set(term_typ, apply_of_tos, state.words_left + len(apply_of_tos)):
#             rest_of_apply_set = apply_set - apply_of_tos
#
#             if len(rest_of_apply_set) <= state.words_left:
#                 possible_sources.update(rest_of_apply_set)
#
#     best_apply_edge_id = None
#     if len(possible_sources) > 0:
#         edge_ids = {self.additional_lexicon.get_id("edge_labels", "APP_"+source) for source in possible_sources}
#         best_apply_edge_id, max_apply_score = get_best_constant(edge_ids, label_scores)
#
#     # Check MODIFY
#     max_modify_score = -np.inf
#     best_modify_edge_id = None
#     if state.words_left - smallest_apply_set > 0:
#         best_modify_edge_id, max_modify_score = get_best_constant(self.modify_ids, label_scores)
#
#     # Apply our choice
#     if max_modify_score > max_apply_score:
#         # MOD
#         return Decision(int(selected_node), self.additional_lexicon.get_str_repr("edge_labels",  best_modify_edge_id), ("",""),"", score=score+max_modify_score)
#     elif max_apply_score > -np.inf:
#         # APP
#         return Decision(int(selected_node), self.additional_lexicon.get_str_repr("edge_labels",  best_apply_edge_id), ("",""),"", score=score+max_apply_score)
#     else:
#         raise ValueError("Could not select action. Bug.")
