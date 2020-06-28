from typing import Tuple, Set

import torch

def make_bool_multipliable(t : torch.Tensor) -> torch.Tensor:
    if t.is_cuda:
        if not (t.dtype == torch.float or t.dtype == torch.float16):
            return t.half() #float 16 is good enough, I hope.
    else:
        if not (t.dtype == torch.int32 or t.dtype == torch.long or t.dtype == torch.int16):
            return t.int()
    return t

def are_eq(a : torch.Tensor, b : torch.Tensor) -> torch.BoolTensor:
    if a.dtype == torch.float or a.dtype == torch.float16 or b.dtype == torch.float or b.dtype == torch.float16:
        return torch.abs(a-b) < 0.00001
    else:
        return a == b


def tensor_or(tensor : torch.BoolTensor, dim : int) -> torch.BoolTensor:
    return tensor.sum(dim=dim) > 0


def index_OR(batched_set : torch.BoolTensor, mapping : torch.BoolTensor) -> torch.BoolTensor:
    """
    batched_set : shape (batch_size, set capacity)
    mapping : shape (set capacity, "constants")
    returns a bool tensor R of shape (batch_size, "constants")
    R[b,c] = True iff \exists l, batched_set[b,l] AND mapping[l,c]
    """
    result = batched_set @ mapping #shape (batch_size, "constants")
    return result > 0

def batched_index_OR(batched_set : torch.BoolTensor, mapping : torch.BoolTensor) -> torch.BoolTensor:
    """
    batched_set : shape (batch_size, set capacity)
    mapping : shape (batch_size, set capacity, "constants")
    returns a bool tensor R of shape (batch_size, "constants")
    R[b,c] = True iff \exists l, batched_set[b,l] AND mapping[b,l,c]
    """
    result = (torch.bmm(batched_set.unsqueeze(1), mapping)).squeeze(1) #shape (batch_size, "lexical types")
    return result > 0


def consistent_with_and_can_finish_now(batched_set: torch.BoolTensor, mapping: torch.BoolTensor, apply_set_exists : torch.BoolTensor,
                                       minimal_apply_set_size: torch.Tensor,
                                       obligatory_apply_set: torch.Tensor) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
    """
    batched_set : shape (batch_size, set capacity)
    mapping : shape (batch_size, set capacity, "lexical types")
    apply_set_exists : shape (batch_size, "lexical types")
    minimal_apply_set_size: shape (batch_size, "lexical types")
    obligatory_apply_set: shape (batch_size, set capacity, "lexical types")

    returns a tuple of bool tensors, both of shape (batch_size, "lexical types")
    1.) R_1[b,l] = True iff R_2[b,l] AND at sum_s[ (batched_set[b,s] AND mapping[b,s,l]) ] >= minimal_apply_set_size[b,l]
    2.) R_2[b,l] = True iff \forall s, batched_set[b,s] -> mapping[b, s, l] AND apply_set_exists[b,l]
    """
    set_size = batched_set.sum(dim=1) #shape (batch_size,)
    #result = torch.einsum("bs, bsl -> bl", batched_set, mapping)
    batched_set_unsq = batched_set.unsqueeze(1)
    result = (torch.bmm(batched_set_unsq, mapping)).squeeze(1) #shape (batch_size, "lexical types")
    r2 = (torch.bmm(batched_set_unsq, obligatory_apply_set)).squeeze(1) #shape (batch_size, "lexical types")
    # number of obligatory sources that we have already.

    #s = obligatory_apply_set.sum(dim=1) #shape (batch_size, "lexical types",), contains the size of the apply set for each lexical type

    consistent = are_eq(result, set_size.unsqueeze(1)) #shape (batch_size, "lexical types")
    # OK but does not take into account that not every pair of types is apply-reachable, so find lexical types that are apply reachable to our
    # set of term types
    consistent &= apply_set_exists

    #can_finish_now = consistent & (result >= minimal_apply_set_size)
    can_finish_now = consistent & (r2 >= minimal_apply_set_size)
    #can_finish_now = consistent & are_eq(result, s)

    return can_finish_now, consistent, r2

def debug_to_set(t : torch.BoolTensor) -> Set[int]:
    return [{ i for i,x in enumerate(batch) if x} for batch in t.numpy()]