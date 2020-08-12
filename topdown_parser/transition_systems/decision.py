import torch
from typing import Optional, Tuple

from dataclasses import dataclass

from topdown_parser.am_algebra import AMType
from topdown_parser.dataset_readers.AdditionalLexicon import AdditionalLexicon


@dataclass(frozen=True)
class Decision:
    position : int
    pop: bool
    label : str
    supertag : Tuple[str, str]
    lexlabel : str
    termtyp : Optional[AMType] = None
    score: float = 0.0

@dataclass(frozen=True)
class DecisionBatch:
    push_tokens: torch.Tensor #shape (batch_size,)
    push_mask: torch.Tensor #shape (batch_size,) for which batch elements do we perform push?
    pop_mask: torch.Tensor #shape (batch_size,) shall we perform a pop operation?

    edge_labels: torch.Tensor #shape (batch_size,)

    constants: torch.Tensor #shape (batch_size,)
    term_types: Optional[torch.Tensor] #shape (batch_size,)
    lex_labels: torch.Tensor
    constant_mask: torch.Tensor

    @staticmethod
    def from_decision(decision : Decision, lexicon: AdditionalLexicon) -> "DecisionBatch":
        return DecisionBatch(torch.tensor([decision.position]),
                             torch.tensor([int(not decision.pop)]),
                             torch.tensor([int(decision.pop)]),
                             torch.tensor([lexicon.get_id("edge_labels", decision.label)]),
                             torch.tensor([lexicon.get_id("constants", "--TYPE--".join(decision.supertag))]),
                             torch.tensor([lexicon.get_id("term_types", str(decision.termtyp)) if decision.termtyp is not None else 0]),
                             torch.tensor([lexicon.get_id("lex_labels", decision.lexlabel)]),
                             torch.tensor([decision.supertag != ("", "")])
                             )

    def to(self, device):
        return DecisionBatch(self.push_tokens.to(device),
                             self.push_mask.to(device),
                             self.pop_mask.to(device),
                             self.edge_labels.to(device),
                             self.constants.to(device),
                             self.term_types.to(device),
                             self.lex_labels.to(device),
                             self.constant_mask.to(device))
