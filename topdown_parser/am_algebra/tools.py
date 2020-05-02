from typing import Optional, Tuple, List, Dict, Set, Iterable

from topdown_parser.dataset_readers.amconll_tools import AMSentence, Entry
from .new_amtypes import AMType, ReadCache, NonAMTypeException
from .tree import Tree

def get_tree_type(sent : AMSentence) -> Optional[AMType]:
    deptree = Tree.from_am_sentence(sent)
    root = sent.get_root()
    if root is None:
        return None

    term_types = get_term_types(deptree, sent)
    return term_types[root]

def get_term_types(deptree : Tree, sent : AMSentence) -> List[Optional[AMType]]:
    cache = ReadCache()
    term_types = [None for _ in sent.words]

    def determine_tree_type(node : Tuple[int, Entry], children : List[Tuple[Optional[AMType],str]]) -> Tuple[Optional[AMType],str]:
        try:
            lextyp = cache.parse_str(node[1].typ)
        except NonAMTypeException:
            return None, node[1].label

        apply_children : Dict[str, AMType] = dict() # maps sources to types of children

        for child_typ, label in children:
            if child_typ is None: # one child is not well-typed
                return None, node[1].label

            if "_" in label:
                source = label.split("_")[1]
                if label.startswith("MOD") and not lextyp.can_be_modified_by(child_typ, source):
                    return None, node[1].label
                elif label.startswith("APP"):
                    apply_children[source] = child_typ
            else:
                if label == "IGNORE":
                    if not child_typ.is_bot:
                        return None, node[1].label

                elif label == "ROOT":
                    if node[1].head == -1:
                        return child_typ, node[1].label
                    else:
                        return None, node[1].label
                else:
                    raise ValueError("Nonsensical edge label: "+label)

        typ = lextyp
        changed = True
        while changed:
            changed = False
            remove = []
            for o in apply_children:
                if typ.can_apply_to(apply_children[o], o):
                    typ = typ.perform_apply(o)
                    remove.append(o)
                    changed = True

            for source in remove:
                del apply_children[source]

        if apply_children == dict():
            term_types[node[0]-1] = typ
            return typ, node[1].label

        return None, node[1].label

    deptree.fold(determine_tree_type)
    return term_types

def is_welltyped(sent : AMSentence) -> bool:
    return get_tree_type(sent) is not None

# def top_down_term_types(sent : AMSentence) -> List[Set[AMType]]:
#     """
#     Returns for every token the set of possible term types when looking from a top-down perspective.
#     :param sent:
#     :return:
#     """
#     term_types = [set() for _ in sent.words]
#     lex_types = [AMType.parse_str(w.typ) for w in sent.words]
#
#     for i,word in enumerate(sent.words):
#         if word.label == "ROOT":
#             term_types[i] = {AMType.parse_str("()")} # empty term type at the root
#
#         elif word.label.startswith("APP_"):
#             # incoming APP_x edge
#             source = word.label.split("_")[1]
#             parent_lex_type = lex_types[word.head-1]
#             term_types[i] = {parent_lex_type.get_request(source)}
#
#         elif word.label.startswith("MOD_"):
#             source = word.label.split("_")[1]
#             max_subtype = lex_types[word.head-1]
#             for subtyp in get_all_subtypes(max_subtype):
#                 subtyp.add_node(source)
#                 term_types[i].add(subtyp)
#
#     return term_types



# class SubtypeCache:
#     def __init__(self, omega : Iterable[AMType]):
#         self.subtypes_of : Dict[AMType, Set[AMType]] = {t : set() for t in omega}
#
#         for t1 in omega:
#             for t2 in omega:
#                 if t1.is_compatible_with(t2): #t1 is subgraph of t2
#                     self.subtypes_of[t2].add(t1)
#
#     def get_subtypes_of(self, t):
#         return self.subtypes_of[t]



