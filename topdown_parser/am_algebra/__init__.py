import pyximport

pyximport.install()

from .new_amtypes import AMType, CombinationCache, ReadCache, NonAMTypeException
from .tree import Tree
from .dag import DiGraph
