import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg


def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        rules = self.grammar.rhs_to_rules
        # Initialization 
        n = len(tokens)
        pi = [[set([]) for j in range(n+1)]for i in range(n+1)]
        for i in range(n):
            for A in rules[(tokens[i],)]:
                pi[i][i+1].add(A[0])
        # main loop
        for length in range(2,n+1):
            for i in range(n-length+1):
                j = i + length
                for k in range(i+1,j):
                    M = set([])
                    Bset,Cset = pi[i][k],pi[k][j]
                    for b in Bset:
                        for c in Cset:
                            for A in rules[(b,c)]:
                                M.add(A[0])
                    for m in M:
                        pi[i][j].add(m)
        # final
        S = self.grammar.startsymbol
        if S in pi[0][-1] :
            return True
        else:
            return False

        
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """  
        rules = self.grammar.rhs_to_rules
        probs = {} 
        table = {}  
         
        # Initialization 
        n = len(tokens)
        for i in range(n+1):
            for j in range(n+1):
                table[(i, j)] = {}
                probs[(i, j)] = {}

        for i in range(n):
            A = rules[(tokens[i],)] # rules that contain word on the rhs
            for a in A:
                table[(i, i+1)][a[0]] = a[1][0]
                probs[(i, i+1)][a[0]] = math.log(a[-1])

        # main loop
        for length in range(2, n+1):
            for i in range(n-length+1):
                j = i + length
                for k in range(i+1, j):
                   # e.g. key = ('ASEARLY', 'PP')  
                   # e.g. value = [('ADJP', ('ASEARLY', 'PP'), 0.0588235294118)]
                   for key, value in rules.items(): 
                        for B in table[(i, k)]:
                            for C in table[(k, j)]:
                                if key[0] == B and key[1] == C: # A -> B C
                                    for a in value:
                                        # product of the three probabilities = sum of the three log probabilities
                                        pi = math.log(a[2]) + probs[(i,k)][B] + probs[(k,j)][C]
                                        # take the maximum 
                                        if a[0] in table[(i,j)]:
                                            # check if that split would produce a higher log probability
                                            if  probs[(i, j)][a[0]] < pi:
                                                # If so, update the entry in the backpointer table & probability table
                                                probs[(i, j)][a[0]] = pi
                                                table[(i, j)][a[0]] = ((key[0],i,k), (key[1],k,j)) 
                                        else:
                                            table[(i,j)][a[0]] = ((key[0],i,k), (key[1],k,j)) 
                                            probs[(i,j)][a[0]] = pi

        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    #print(chart[(i,j)])
    if j-i == 1:
        return (nt,chart[(i, j)][nt])
    
    a, b = chart[(i, j)][nt]
    #print(a,b)
    return [nt, get_tree(chart, a[1], a[2], a[0]), get_tree(chart, b[1], b[2], b[0])]


 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.'] 
        print(parser.is_in_language(toks))
        table,probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)
        tree = get_tree(table, 0, len(toks), grammar.startsymbol)
        
