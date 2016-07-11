import numpy as np
from WFA import WFA

class SimpleTeacher:
    """ A class to define a simple WFA, which computes length(w)+1 on input w. 
    Equivalence queries check certain strings of length up to self.size.
    """
    def __init__(self):
        self.size = 100
        self.tolerance = 1e-7
        
    def MQ(self,w):
        return len(w)+1 # note: the alg assumes that f(eps)!=0. Otherwise, translate.
        
    def EQ(self,h):
        for i in range(self.size):
            s = '0'*(i/2) + '1'*(i-(i/2))
            # if the solver uses least squares, allow difference up to tolerance
            # if(abs(h.compute(s)-(i+1))>self.tolerance):
            if(h.compute(s)!=i+1):
                return (False,s)
        return(True,'')

# alphabet
Sigma = ['0','1']
eps = ""

# P, PSigma label rows while S labels columns
# rows is a dict with words as keys and the corresponding row indices for H as values
def reconstructWFA(H,P,S,rows): 
    """ Reconstruct a WFA computing a function from a complete minimal block of its Hankel matrix """
    
    n = len(P)
    rowsP = [rows[p] for p in P.keys()]
    rowsPSigma = [rows[p+sig] for p in P.keys() for sig in Sigma]
    # colsS = [cols[s] for s in S]
    
    # blocks
    HB = H[rowsP,:][:,S.values()]
    HBSigma = H[rowsP + rowsPSigma,:][:,S.values()]

    # starting vector
    alpha = np.zeros(n); alpha[0] = 1
    # ending vector, assume the first element of P and S is epsilon
    eta = HB[:,0]
        
    # check completeness
    # P[0] === S[0] === ''
    # rank(HBSigma) === rank(HB)
    
    # check minimality
    # rank(HB) === n
    
    M = {}
    for sig in Sigma:
        # solve the matrix equation M_sig * HB = H_sig, which is possible when B is complete
        # transpose to fit into np syntax: HB^T * M_sig^T = H_sig^T
        Hsig = H[[rows[p+sig] for p in P.keys()],:][:,S.values()]
        # M[sig] = np.linalg.lstsq(HB.transpose(),Hsig.transpose())[0].transpose()
        M[sig] = np.linalg.solve(HB.transpose(),Hsig.transpose()).transpose()
    
    return WFA(n,alpha,eta,M)

def learn(teacher):
    """ Learn a WFA from a teacher oracle, using MQs and EQs. 
        We assume the function f to be learned is not zero at epsilon.
    """
    # init
    P = {eps:0}
    S = {eps:0}
    H = np.array([[teacher.MQ(eps)]])
    rows = {eps:0}
    # cols = {eps:0}
    
    # add PSigma rows
    H = fillSigma(P,S,H,rows,teacher)
    print "Sigma rows filled."
    print(H)
    # the initial table is complete and minimal by construction, if f(eps)!=0
    learningComplete = False
    print("Starting learning.")
    while(not learningComplete):
        # as block is complete and minimal, compute the hypothesis automaton
        hypothesis = reconstructWFA(H,P,S,rows)
        print "Hypothesis constructed."
        print [hypothesis.compute(p) for p in P.keys()]
        (b,z) = teacher.EQ(hypothesis) 
        if(b): 
            learningComplete = True
        else:
            # process counterexample z
            print "Counterexample received: ",z
            v = processCounterexample(z,P.keys())
            print "Processed suffix: ",v
            S = addSuffixes(S,v)
            print "Suffixes of v added to S."
            print S
            H = updateH(H,P,S,rows,teacher)
            print H
            # check completeness
            HB = H[P.values(),:] # rows labelled by P
            #print(H)
            #print('---')
            #print(HB)
            rankFull = np.linalg.matrix_rank(H)
            rankB = np.linalg.matrix_rank(HB)
            #print(rankFull)
            #print(rankB)
            if(rankB!=rankFull):
                print 'Block is not complete: rank(HB) = {0}, rank(H) = {1}'.format(rankB,rankFull)
                (P,H) = complete(P,S,H,rows,teacher,HB,rankFull,rankB)
                print 'Block complete.'
                print(H)
                print(P.keys())
                
            # block should be minimal by construction, since we only add a prefix if it increases rank, but we can check this here
            HB = H[P.values(),:] # rows labelled by P
            rankFull = np.linalg.matrix_rank(H)
            rankB = np.linalg.matrix_rank(HB)
            print "Ranks: {0},{1}".format(rankB,rankFull)
            if(rankB!=len(P)):
                print 'Block is not minimal: rank(HB) = {0}, |P| = {1}'.format(rankB,len(P))

    print 'Learning complete.'
    return hypothesis
    
def fillSigma(P,S,H,rows,teacher):
    """ Fill in the block and row indices so that
        the rows are labelled by P *and* PSigma
        the columns are labelled by S 
        We assume that it is labelled by P,S to begin with
    """
    print 'Filling rows labelled PSigma.'
    rowPos = len(rows)
    for p in P.keys():
        for sig in Sigma:
            if not(p+sig in rows):
                newrow = [teacher.MQ(p+sig+s) for s in S.keys()]
                H = np.append(H,[np.array(newrow)],axis=0)
                rows[p+sig] = rowPos
                rowPos+=1
    return H
def complete(P,S,H,rows,teacher,HB,rankFull,rankB):
    """ Complete the given block (with rows labelled by P U PSigma) by finding
        rows labelled by PSigma that are independent of the rows labelled by P,
        extending P to include these, and filling the block again. 
    """
    for p in P.keys():
        if(rankFull == rankB): break;
        for sig in Sigma:
            if(rankFull == rankB): break;
            rowpsig = H[rows[p+sig]]
            Hpsig = np.append(HB,[rowpsig],axis=0)
            rankpsig = np.linalg.matrix_rank(Hpsig)
            if(rankpsig != rankB):
                P[p+sig]=rows[p+sig]
                H = fillSigma(P,S,H,rows,teacher)
                rankB = rankpsig
    return (P,H) 

def processCounterexample(z,prefixes):
    # find decomposition z = uav where u is the longest prefix of z in prefixes
    # note z!=eps as this is correct on to start with
    for i in range(len(z)):
        # u = z[0..len-i-1), a = z[len-i-1], v = z[len-i..len)
        u = z[0:len(z)-i-1]
        if(u in prefixes):
            a = z[len(z)-i-1]
            v = z[len(z)-i:]    
            print "Decomposition found: {0} = {1}+{2}+{3}".format(z,u,a,v)
            return v
    # when i = len(z)-1 then we have u = eps which is always a prefix

def addSuffixes(S,v):
    count = len(S)
    for i in range(len(v)):
        # add suffix v[i..len) to S
        s = v[i:]
        if(not(s in S)):
            S[s] = count
            count+=1  
    return S

def updateH(H,P,S,rows,teacher):        
    # extend H to the new columns of S using MQ      
    newH = np.zeros((H.shape[0],len(S)))  
    newH[0:H.shape[0],0:H.shape[1]] = H
    for s,sCol in S.items():
        if(sCol >= H.shape[1]): # column not already in H
            for p,pRow in P.items():
                newH[pRow][sCol] = teacher.MQ(p+s)
                for sig in Sigma:
                    newH[rows[p+sig]][sCol] = teacher.MQ(p+sig+s)
    return newH
                             
answer = learn(SimpleTeacher())   
answer.display()
