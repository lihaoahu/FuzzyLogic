from fuzzy_logic.interval_type_2.fuzzy_sets import Chen_TrapezoidalIT2FS as TIT2FS
from numpy import array,sum

class TIT2F_VIKOR:

    def __init__(self,decision_matrix,weights,theta=0.5):
        self.D = array(decision_matrix)
        self.W = array(weights)
        self.theta = theta
        self.n,self.m = self.D.shape
        
    @staticmethod
    def distance(A,B):
        return ((
                abs(A.lmf_left-B.lmf_left) + 2*abs(A.lmf_left_apex-B.lmf_left_apex) + 2*abs(A.lmf_right_apex-B.lmf_right_apex) + abs(A.lmf_right-B.lmf_right)
            )*min([A.lmf_right_height,B.lmf_right_height])**2 + (
                abs(A.umf_left-B.umf_left) + 2*abs(A.umf_left_apex-B.umf_left_apex) + 2*abs(A.umf_right_apex-B.umf_right_apex) + abs(A.umf_right-B.umf_right)
            )*min([A.umf_right_height,B.umf_right_height])**2)/12

    
    def evaluate(self):
        positive = []
        negtive = []
        for j in range(self.m):
            negtive.append(TIT2FS(
                min([self.D[i,j].umf_left for i in range(self.n)]),
                min([self.D[i,j].umf_left_apex for i in range(self.n)]),
                min([self.D[i,j].umf_right_apex for i in range(self.n)]),
                min([self.D[i,j].umf_right for i in range(self.n)]),
                min([self.D[i,j].umf_right_height for i in range(self.n)]),
                min([self.D[i,j].lmf_left for i in range(self.n)]),
                min([self.D[i,j].lmf_left_apex for i in range(self.n)]),
                min([self.D[i,j].lmf_right_apex for i in range(self.n)]),
                min([self.D[i,j].lmf_right for i in range(self.n)]),
                min([self.D[i,j].lmf_right_height for i in range(self.n)])
            ))
            positive.append(TIT2FS(
                max([self.D[i,j].umf_left for i in range(self.n)]),
                max([self.D[i,j].umf_left_apex for i in range(self.n)]),
                max([self.D[i,j].umf_right_apex for i in range(self.n)]),
                max([self.D[i,j].umf_right for i in range(self.n)]),
                max([self.D[i,j].umf_right_height for i in range(self.n)]),
                max([self.D[i,j].lmf_left for i in range(self.n)]),
                max([self.D[i,j].lmf_left_apex for i in range(self.n)]),
                max([self.D[i,j].lmf_right_apex for i in range(self.n)]),
                max([self.D[i,j].lmf_right for i in range(self.n)]),
                max([self.D[i,j].lmf_right_height for i in range(self.n)])
            ))
        S = []
        R = []
        G = []
        for i in range(self.n):
            wd = [self.W[j]*self.distance(positive[j],self.D[i,j])/self.distance(positive[j],negtive[j]) for j in range(self.m)]
            S.append(sum(wd))
            R.append(max(wd))
        for i in range(self.n):
            G.append(self.theta*(S[i]-min(S))/(max(S)-min(S))+(1-self.theta)*(R[i]-min(R))/(max(R)-min(R)))
        self.S,self.R,self.G = S,R,G