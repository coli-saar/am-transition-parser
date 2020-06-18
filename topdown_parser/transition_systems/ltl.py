from typing import List, Iterable, Optional, Any, Dict

import torch

from topdown_parser.am_algebra.tree import Tree
from topdown_parser.dataset_readers.AdditionalLexicon import AdditionalLexicon
from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.datastructures.list_of_list import BatchedListofList
from topdown_parser.datastructures.stack import BatchedStack
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
        :param applyset: shape (batch_size,), belongs to currently active nodes
        """
        super(LTLState, self).__init__(decoder_state, sentences, stack, children, heads, edge_labels, constants, None, lex_labels, lexicon)
        self.lex_types = lex_types
        self.applyset = applyset

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
                                     torch.zeros(batch_size, max_len, device=device, dtype=torch.long), # term_types
                                     torch.zeros(batch_size, max_len, device=device, dtype=torch.long), #lex labels
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
        lex_labels = scores["lex_labels"]

        return DecisionBatch(selected_nodes, push_mask, pop_mask, edge_labels, constants, None, lex_labels, pop_mask)




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

        pop_mask = decision_batch.pop_mask #shape (batch_size,)
        active_children = state.children.lol[range_batch_size, next_active_nodes] #shape (batch_size, max. number of children)
        push_all_children_mask = (active_children != 0).long() #shape (batch_size, max. number of children)
        push_all_children_mask *= pop_mask.unsqueeze(1) # only push those children where we will pop the current node from the top of the stack.

        state.stack.pop_and_push_multiple(active_children, decision_batch.pop_mask, push_all_children_mask, reverse=self.reverse_push_actions)
