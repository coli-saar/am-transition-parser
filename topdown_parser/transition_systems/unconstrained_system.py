from typing import Dict, Union, List, Set

import torch

from topdown_parser.dataset_readers.AdditionalLexicon import AdditionalLexicon
from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.nn.EdgeLabelModel import EdgeLabelModel
from topdown_parser.transition_systems.parsing_state import CommonParsingState
from topdown_parser.transition_systems.transition_system import Decision, TransitionSystem
from topdown_parser.transition_systems.utils import single_score_to_selection


class UnconstrainedTransitionSystem(TransitionSystem):
    """
    Mix-in for DFS and DFS-children first for making decisions.
    """

    def __init__(self, additional_lexicon: AdditionalLexicon, pop_with_0: bool):
        super().__init__(additional_lexicon)
        self.pop_with_0 = pop_with_0
        self.additional_lexicon = additional_lexicon

    def predict_supertag_from_tos(self) -> bool:
        return True

    def make_decision(self, scores: Dict[str, torch.Tensor], label_model: EdgeLabelModel, state : CommonParsingState) -> Decision:
        # Select node:
        child_scores = scores["children_scores"].detach().cpu() # shape (input_seq_len)
        #Cannot select nodes that we have visited already.

        for seen in state.seen:
            if self.pop_with_0 and seen == 0:
                pass
            elif not self.pop_with_0 and seen == state.active_node:
                pass
            else:
                child_scores[seen] = -10e10

        score = 0.0
        s, selected_node = torch.max(child_scores, dim=0)
        score += s
        s, selected_label = single_score_to_selection(scores, self.additional_lexicon, "edge_labels")
        score += s
        s, selected_supertag = single_score_to_selection(scores, self.additional_lexicon, "constants")
        score += s
        s, selected_lex_label = single_score_to_selection(scores, self.additional_lexicon, "lex_labels")
        score += s

        return Decision(int(selected_node), selected_label, AMSentence.split_supertag(selected_supertag), selected_lex_label, score=score)

    def top_k_decision(self, scores: Dict[str, torch.Tensor], encoder_state : Dict[str,torch.Tensor],
                       label_model: EdgeLabelModel, state : CommonParsingState, k : int) -> List[Decision]:
        # Select node:
        child_scores = scores["children_scores"] # shape (input_seq_len)
        #Cannot select nodes that we have visited already (except if not pop with 0 and currently active, then we can close).
        forbidden = 0
        for seen in state.seen:
            if self.pop_with_0 and seen == 0:
                pass
            elif not self.pop_with_0 and seen == state.active_node:
                pass
            else:
                child_scores[seen] = -5e10
                forbidden += 1

        at_most_k = min(k, len(state.sentence)+1-forbidden) #don't let beam search explore things that are not well-formed.
        children_scores, children = torch.sort(child_scores, descending=True)
        children_scores = children_scores[:at_most_k] #shape (at_most_k)
        children = children[:at_most_k] #shape (at_most_k)
        # Now have k best children
        encoded_input = encoder_state["encoded_input"].repeat((at_most_k, 1, 1)) #shape (at_most_k, seq_len, encoder dim)
        input_mask = encoder_state["input_mask"].repeat((at_most_k, 1, 1)) #shape (at_most_k, seq_len, encoder_dim)
        label_model.set_input(encoded_input,input_mask)
        decoder_hidden = encoder_state["decoder_hidden"] #shape (decoder embedding)
        decoder_hidden = decoder_hidden.repeat((at_most_k, 1)) #shape (at_most_k, decoder embedding)
        label_scores = label_model.edge_label_scores(children, decoder_hidden) #shape (at_most_k, label vocab dim)

        label_scores, best_labels = torch.max(label_scores, dim=1)

        children = children.cpu()
        children_scores = children_scores.cpu()
        label_scores = label_scores.cpu()
        best_labels = best_labels.cpu().numpy()

        #Constants, lex label:
        score_constant, selected_supertag = single_score_to_selection(scores, self.additional_lexicon, "constants")
        score_lex_label, selected_lex_label = single_score_to_selection(scores, self.additional_lexicon, "lex_labels")

        return [Decision(int(children[i]),self.additional_lexicon.get_str_repr("edge_labels",best_labels[i]),
                         AMSentence.split_supertag(selected_supertag), selected_lex_label,score=score_lex_label + score_constant + label_scores[i] + children_scores[i])
                for i in range(at_most_k)]

    def assumes_greedy_ok(self) -> Set[str]:
        """
        The dictionary keys of the context provider which we make greedy decisions on in top_k_decisions
        because we assume these choices won't impact future scores.
        :return:
        """
        return {"children_labels", "lexical_types"}