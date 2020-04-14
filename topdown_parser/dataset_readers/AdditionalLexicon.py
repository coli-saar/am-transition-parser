from typing import Tuple, Dict, List

from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError


class Lexicon:

    def __init__(self, file_name : str):
        self.s2i = dict()
        self.i2s = []
        with open(file_name) as fil:
            for i,line in enumerate(fil):
                line = line.rstrip("\n")
                self.i2s.append(line)
                self.s2i[line] = i+1

    def vocab_size(self) -> int:
        return len(self.i2s)+1 # UNK symbol

    def get_id(self, s : str):
        if s in self.s2i:
            return self.s2i[s]
        return 0



class AdditionalLexicon(Registrable):
    """
    A class with a lexicon for things like
     - edge labels
    this is needed because when the context is gathered in the DatasetReader, the mapping
    between these things and ids is not performed yet.
    """
    POSSIBLE_KEYS = {"edge_labels", "lexical_types"}

    def __init__(self, sublexica : Dict[str, str]):
        super().__init__()

        if not set(sublexica.keys()).issubset(AdditionalLexicon.POSSIBLE_KEYS):
            raise ConfigurationError(f"Unkown keys used: {sublexica.keys()}, I only know {AdditionalLexicon.POSSIBLE_KEYS}")

        self.sublexica = { name : Lexicon(path) for name, path  in sublexica.items()}
