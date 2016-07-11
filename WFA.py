import numpy as np
class WFA:
    """ Class for weighted finite automata over the reals. """
    
    def __init__(self, n, alpha, eta, M):
        """ Initialise a weighted finite automaton over the reals. 
            n is the number of states
            alpha is the starting vector
            eta is the ending vector
            M is a dictionary, with the alphabet as the set of keys,
            and the corresponding transition matrices as the values.
        """
        self.n = n
        if(len(M)==0): 
            raise ValueError("Must have at least one matrix.")
        self.sigma = M.keys()
        self.alpha = alpha
        self.eta = eta
        self.M = M # dict from letters of sigma to the corresponding transition matrices
        
    def compute(self, w):
        """ Compute the output on the input string w """
        current = self.alpha.transpose()
        for i in range(len(w)):
            # print(i)
            # print(current)
            # print(self.M[w[i]])
            current = current.dot(self.M[w[i]])
        return current.dot(self.eta)
    
    def display(self):
        print "---\nDisplaying a weighted finite automaton."
        print "Starting vector alpha:"
        print self.alpha
        print "Ending vector eta:"
        print self.eta
        print "Transition matrices:"
        for sig in self.sigma:
            print "Matrix for {0}: \n{1}".format(sig, self.M[sig])
        print "---"
        
    
#aut = WFA(2,np.array([0,1]),np.array([1,0]),{'0':np.array([[1,0],[1,1]]),'1':np.array([[1,0],[0,1]])})
#print aut.compute('0000110')   
 