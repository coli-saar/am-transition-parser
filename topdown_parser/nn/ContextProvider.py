from typing import Dict, Any, List

import torch
from allennlp.common import Registrable
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
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

    def compute_context(self, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Takes the context and distills it into a vector that can then be combined in forward with the representation of the current node.
        :param state:
        :param context:
        :return:
        """
        raise NotImplementedError()



@ContextProvider.register("no_context")
class NoContextProvider(ContextProvider):

    def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, Any]) -> torch.Tensor:
        return current_node

@ContextProvider.register("parent")
class ParentContextProvider(ContextProvider):
    """
    Add parent information in Ma et al. 2018 style
    """

    def compute_context(self, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = state["encoded_input"].shape[0]

        parents = context["parents"] # shape (batch_size,)

        encoded_parents = state["encoded_input"][range(batch_size), parents] # shape (batch_size, encoder dim)

        #mask = parents == 0 # which parents are 0? Skip those
        #encoded_parents=  mask.unsqueeze(1) * encoded_parents

        return encoded_parents

    def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, Any]) -> torch.Tensor:

        return current_node + self.compute_context(state, context)


class MostRecent(ContextProvider):
    """
    Add information about most recent sibling/child.
    """

    def __init__(self, context_key : str, mask_key : str):
        super().__init__()
        self.context_key = context_key
        self.mask_key = mask_key

    def compute_context(self, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
        # For the sake of the example, let's say we're looking for siblings
        siblings = context[self.context_key] #shape (batch_size, max_num_siblings)
        batch_size, _ = siblings.shape

        sibling_mask = context[self.mask_key] # (batch_size, max_num_siblings)

        number_of_siblings = get_lengths_from_binary_sequence_mask(sibling_mask) # (batch_size,)

        most_recent_sibling = siblings[range(batch_size), number_of_siblings-1] # shape (batch_size,)

        encoded_sibling = state["encoded_input"][range(batch_size), most_recent_sibling] # shape (batch_size, encoder_dim)

        # Some nodes don't have siblings, mask them out:
        encoded_sibling = (number_of_siblings != 0).unsqueeze(1) * encoded_sibling #shape (batch_size, encoder_dim)
        return encoded_sibling

    def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:

        return current_node + self.compute_context(state, context)


@ContextProvider.register("most-recent-sibling")
class SiblingContextProvider(ContextProvider):
    """
    Add information about most recent sibling, like Ma et al.
    """

    def __init__(self):
        super().__init__()
        self.most_recent = MostRecent("siblings", "siblings_mask")

    def compute_context(self, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.most_recent.compute_context(state, context)

    def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:

        return current_node + self.compute_context(state, context)


@ContextProvider.register("most-recent-child")
class SiblingContextProvider(ContextProvider):
    """
    Add information about most recent sibling, like Ma et al.
    """

    def __init__(self):
        super().__init__()
        self.most_recent = MostRecent("children", "children_mask")

    def compute_context(self, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.most_recent.compute_context(state, context)

    def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:

        return current_node + self.compute_context(state, context)



@ContextProvider.register("sum")
class SumContextProver(ContextProvider):
    """
    Add information about most recent sibling, like Ma et al.
    """

    def __init__(self, providers : List[ContextProvider]):
        super().__init__()
        self.providers = providers

    def forward(self, current_node : torch.Tensor, state : Dict[str, torch.Tensor], context : Dict[str, torch.Tensor]) -> torch.Tensor:
        r = current_node

        for provider in self.providers:
            r = r + provider.compute_context(state, context)

        return r