from typing import List

from topdown_parser.dataset_readers.amconll_tools import parse_amconll, AMSentence, Entry


class Tree:
    """
    A simple tree class
    """
    def __init__(self, node, children):
        self.node = node
        self.children = children
        
    def add_child(self,child):
        self.children.append(child)
    
    @staticmethod
    def from_heads(heads, conll_sentence : List):
        
        def parse(i):
            mother = Tree((i,conll_sentence[i]),[])
            for j in range(len(heads)):
                if heads[j] == i:
                    mother.add_child(parse(j))
            return mother
        
        return parse(0) #articial root is at position 0

    @staticmethod
    def from_am_sentence(s : AMSentence):
        root = "*root*"
        root_entry = Entry(root,"_",root, root, root, "_", "_", "_", -1, root, True, None)
        return Tree.from_heads([-1] + s.get_heads(),[root_entry] + s.words)
    
    def fold(self, f):
        """
        Folding on trees: f takes a node a list of things that f produces
        Returns a single thing (for instance a Tree or a number)
        """
        if len(self.children) == 0:
            return f(self.node,[])
        return f(self.node,[c.fold(f) for c in self.children])

    def fold_double(self, f):
        """
        Folding on trees: f takes a node a list of things that f produces
        Returns a single thing (for instance a Tree or a number)
        """
        if len(self.children) == 0:
            return f(self.node,[],[])
        return f(self.node,self.children,[c.fold_double(f) for c in self.children])
        
    def map(self,f):
        """
            applies f to all nodes of the tree. changes the tree
        """
        self.node = f(self.node)
        for c in self.children:
            c.map(f)
    
    def size(self):
        if len(self.children) == 0:
            return 1
        return 1+sum(c.size() for c in self.children)

    def max_arity(self):
        if len(self.children) == 0:
            return 0
        return max(len(self.children), max(c.max_arity() for c in self.children))
        
    def postorder(self):
        if self.children == []:
            yield self
        else:
            for c in self.children:
                for x in c.postorder():
                    yield x
            yield self

    def _to_str(self,depth=0):
        if len(self.children) == 0:
            return 4*depth*" "+str(self.node)
        return 3*depth*" "+"["+str(self.node)+"\n {}]".format("\n".join( c._to_str(depth+1) for c in self.children))
    
    def __str__(self):
        return self._to_str()

    def __repr__(self):
        if len(self.children) == 0:
            return "("+str(self.node) +")"
        return "({} {})".format(str(self.node)," ".join(c.__repr__() for c in self.children))

def weird_order(tree):
    agenda = list(reversed(tree.children))
    while agenda:
        current_node = agenda.pop()
        yield current_node
        for child in current_node.children:
            yield child
            agenda.extend(reversed(child.children))
        yield current_node




if __name__ == "__main__":


    with open("data/tratz/gold-dev/gold-dev.amconll") as f:
        sentences = list(parse_amconll(f))
    s = sentences[100]
    print(s)
    print(s.get_heads())

    t = Tree.from_am_sentence(s)

    print(list(x.node[0] for x in weird_order(t)))