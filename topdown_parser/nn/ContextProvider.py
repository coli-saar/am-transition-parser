from typing import Dict, Any

import torch
from allennlp.common import Registrable
from torch.nn import Module


class ContextProvider(Module, Registrable):
    """
    Takes the context extracted by gather_context() in the transition system
    and computes a fixed vector that is used for as input for the edge model and
    is the input to the decoder.
    """


    def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        :param current_node: tensor of shape (batch_size, encoder dim) with representation of active nodes.
        :param state: contains keys "encoded_input", which contains the entire input in a tensor of shape (batch_size, input_seq_len, encoder dim)
        :param context: provided by gather_context a dictionary with values of shape (batch_size, *) with additional dimensions for the current time step.
        :return: of shape (batch_size, decoder_dim)
        """
        raise NotImplementedError()



@ContextProvider.register("no_context")
class NoContextProvider(ContextProvider):

    def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, Any]) -> torch.Tensor:
        return current_node

@ContextProvider.register("parents")
class ParentContextProvider(ContextProvider):
    """
    Add parent information in Ma et al. 2018 style
    """

    def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, Any]) -> torch.Tensor:
        batch_size = state["encoded_input"].shape[0]

        parents = context["parents"] # shape (batch_size,)
        #mask = parents == 0 # which parents are 0? Skip those
        encoded_parents = state["encoded_input"][range(batch_size), parents] # shape (batch_size, encoder dim)

        return current_node + encoded_parents
        #return current_node + mask.unsqueeze(1) * encoded_parents

