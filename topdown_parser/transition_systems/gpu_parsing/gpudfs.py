from typing import List, Iterable, Optional, Tuple, Dict, Any

import torch

from topdown_parser.am_algebra.tree import Tree
from topdown_parser.dataset_readers.AdditionalLexicon import AdditionalLexicon
from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.transition_systems.gpu_parsing.parsing_state import BatchedParsingState, BatchedStack, \
    BatchedListofList
from topdown_parser.transition_systems.gpu_parsing.transition_system import GPUTransitionSystem, Decision, DecisionBatch


class GPUDFSState(BatchedParsingState):

    def is_complete(self) -> torch.Tensor:
        return torch.all(self.stack.is_empty())




@GPUTransitionSystem.register("dfs")
class GPUDFS(GPUTransitionSystem):

    def __init__(self, children_order: str, pop_with_0: bool, additional_lexicon : AdditionalLexicon):
        """
        Select children_order : "LR" (left to right) or "IO" (inside-out, recommended by Ma et al.)
        """
        super().__init__(additional_lexicon)
        self.pop_with_0 = pop_with_0
        assert children_order in ["LR", "IO"], "unknown children order"

        self.children_order = children_order

    def guarantees_well_typedness(self) -> bool:
        return False

    def get_unconstrained_version(self) -> "GPUTransitionSystem":
        """
        Return an unconstrained version that does not do type checking.
        :return:
        """
        return self

    def _construct_seq(self, tree: Tree, is_first_child : bool, parent_type : Tuple[str, str], parent_lex_label : str) -> List[Decision]:
        own_position = tree.node[0]
        to_left = []
        to_right = []
        for child in tree.children:
            if child.node[1].label == "IGNORE":
                continue

            if child.node[0] < own_position:
                to_left.append(child)
            else:
                to_right.append(child)

        if is_first_child:
            beginning = [Decision(own_position, False, tree.node[1].label, parent_type, parent_lex_label)]
        else:
            beginning = [Decision(own_position, False, tree.node[1].label, ("", ""), "")]

        if self.children_order == "LR":
            children = to_left + to_right
        elif self.children_order == "IO":
            children = list(reversed(to_left)) + to_right
        else:
            raise ValueError("Unknown children order: " + self.children_order)

        ret = beginning
        for i, child in enumerate(children):
            ret.extend(self._construct_seq(child, i == 0, (tree.node[1].fragment, tree.node[1].typ), tree.node[1].lexlabel))

        last_position = 0 if self.pop_with_0 else own_position
        if len(tree.children) == 0:
            #This subtree has no children, thus also no first child at which we would determine the type of the parent
            #Let's determine the type now.
            last_decision = Decision(last_position, True, "", (tree.node[1].fragment, tree.node[1].typ),
                                     tree.node[1].lexlabel)
        else:
            last_decision = Decision(last_position, True, "", ("",""), "")
        ret.append(last_decision)
        return ret


    def get_order(self, sentence: AMSentence) -> Iterable[Decision]:
        t = Tree.from_am_sentence(sentence)
        r = self._construct_seq(t, False, ("",""),"")
        return r

    def check_correct(self, gold_sentence : AMSentence, predicted : AMSentence) -> bool:
        return all(x.head == y.head for x, y in zip(gold_sentence, predicted)) and \
               all(x.label == y.label for x, y in zip(gold_sentence, predicted)) and \
               all(x.fragment == y.fragment and x.typ == y.typ for x, y in zip(gold_sentence, predicted))

    def initial_state(self, sentences : List[AMSentence], decoder_state : Any, device: Optional[int] = None) -> GPUDFSState:
        max_len = max(len(s) for s in sentences)+1
        batch_size = len(sentences)
        stack = BatchedStack(batch_size, max_len+2, device=device)
        stack.push(torch.zeros(batch_size, dtype=torch.long, device=device), torch.ones(batch_size, dtype=torch.long, device=device))
        return GPUDFSState(decoder_state, sentences, stack,
                           BatchedListofList(batch_size, max_len, max_len, device=device),
                           torch.zeros(batch_size, max_len, device=device, dtype=torch.long) - 1,  #heads
                           torch.zeros(batch_size, max_len, device=device, dtype=torch.long),  #labels
                           torch.zeros(batch_size, max_len, device=device, dtype=torch.long) - 1,  #constants
                           torch.zeros(batch_size, max_len, device=device, dtype=torch.long),  # term_types
                           torch.zeros(batch_size, max_len, device=device, dtype=torch.long),  #lex labels
                           self.additional_lexicon)
        # decoder_state: Any
        # sentences : List[AMSentence]
        # stack: BatchedStack
        # children : BatchedListofList #shape (batch_size, input_seq_len, input_seq_len)
        # heads: torch.Tensor #shape (batch_size, input_seq_len) with 1-based id of parents, TO BE INITIALIZED WITH -1
        # edge_labels : torch.Tensor #shape (batch_size, input_seq_len) with the id of the incoming edge for each token
        # constants : torch.Tensor #shape (batch_size, input_seq_len)
        # lex_labels : torch.Tensor #shape (batch_size, input_seq_len)
        # lexicon : AdditionalLexicon

    def make_decision(self, scores: Dict[str, torch.Tensor], state : BatchedParsingState) -> DecisionBatch:
        children_scores = scores["children_scores"] #shape (batch_size, input_seq_len)
        mask = state.parent_mask() #shape (batch_size, input_seq_len)
        depth = state.stack.depth() #shape (batch_size,)
        active_nodes = state.stack.peek()
        if self.pop_with_0:
            mask[state.stack.batch_range, 0] *= (depth > 0)
        else:
            mask[state.stack.batch_range, active_nodes] *= (depth > 0)

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

        edge_labels = torch.argmax(scores["all_labels_scores"][state.stack.batch_range, selected_nodes], 1)
        constants = torch.argmax(scores["constants_scores"], 1)
        lex_labels = scores["lex_labels"]  # torch.argmax(scores["lex_labels_scores"], 1)
        term_types = torch.argmax(scores["term_types_scores"], 1)

        constant_mask = state.constant_mask()[state.stack.batch_range, active_nodes]
        constant_mask *= not_done
        return DecisionBatch(selected_nodes, push_mask, pop_mask, edge_labels, constants, term_types, lex_labels, constant_mask)

    def step(self, state: BatchedParsingState, decision_batch: DecisionBatch) -> None:
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
        inverse_constant_mask = (1-decision_batch.constant_mask.long())
        state.constants[range_batch_size, next_active_nodes] = inverse_constant_mask * state.constants[range_batch_size, next_active_nodes] + decision_batch.constant_mask * decision_batch.constants
        state.lex_labels[range_batch_size, next_active_nodes] = inverse_constant_mask*state.lex_labels[range_batch_size, next_active_nodes] + decision_batch.constant_mask * decision_batch.lex_labels

        state.stack.push(decision_batch.push_tokens, decision_batch.push_mask.bool())
        state.stack.pop_wo_peek(decision_batch.pop_mask.bool())
