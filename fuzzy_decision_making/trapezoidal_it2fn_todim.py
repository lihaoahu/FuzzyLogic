from fuzzy_logic.interval_type_2.fuzzy_sets import Chen_TrapezoidalIT2FS as TIT2FS
from numpy import array

class TIT2F_TODIM:

    def __init__(self,decision_matrix,weights,theta=1,lamb=0.5):
        self.D = array(decision_matrix)
        self.W = array(weights)
        self.theta = theta
        self.lamb = lamb
        self.n,self.m = self.D.shape


    def dominance(self,j,i,k):
        rij = self.D[i,j]
        rkj = self.D[k,j]
        rank_rij = rij.rank_value()
        rank_rkj = rkj.rank_value()
        if rank_rij - rank_rkj > 0:
            return (self.W[j]*rij.distance(rkj))**(1/2)
        elif rank_rij == rank_rkj:
            return 0
        else:
            return -1/self.theta*(sum([self.D[k,crit].distance(self.D[i,crit]) for crit in range(self.m)])/self.W[j])**(1/2)
    
    def evaluate(self):
        self.Dom = []
        for j in range(self.m):
            Dom_j = []
            for i in range(self.n):
                Dom_ji = []
                for k in range(self.n):
                    Dom_jik = self.dominance(j,i,k)
                    Dom_ji.append(Dom_jik)
                Dom_j.append(Dom_ji)
            self.Dom.append(Dom_j)
        self.Dom = array(self.Dom)
        
    

