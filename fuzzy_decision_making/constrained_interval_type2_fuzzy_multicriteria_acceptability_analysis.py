from numpy import vectorize,array
from RankOfFuzzyNumbers.fuzzy_rank_acceptability_analysis import FRAA
from fuzzy_logic.constrained_interval_type_2.fuzzy_sets import TriangularConstrainedIntervalType2FuzzySet

def singly_sample(x):
    return x.singly_sample()

singly_sample = vectorize(singly_sample)

class TriCT2F_matrix:
    def __init__(self, mat) -> None:
        self.mat = array(mat)
        n, _ = self.mat.shape
        self.n_alt = n
    
    def singly_sample(self):
        return singly_sample(self.mat)

    def sample(self,N):
        return [self.singly_sample() for _ in range(N)]

class TriCT2FMAA:
    def __init__(self,TriCT2F_mat):
        self.mat = TriCT2F_mat
        # self.weights = weights
        assert isinstance(TriCT2F_mat,TriCT2F_matrix)
        # assert isinstance(weights,TriCT2F_matrix)
        self.samples = None
    
    def sample(self,N):
        # self.samples = list(zip(self.mat.sample(N),self.weights.sample(N)))
        self.samples = list(self.mat.sample(N))
    
    def evaluate(self):
        if self.samples:
            pass
        else:
            self.sample(100)
        

