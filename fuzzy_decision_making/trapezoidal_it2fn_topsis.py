from fuzzy_logic.interval_type_2.fuzzy_sets import Chen_TrapezoidalIT2FS as TIT2FS
from numpy import array,sum

class TIT2F_TOPSIS:

    def __init__(self,decision_matrix,weights):
        self.D = array(decision_matrix)
        self.W = array(weights)
        self.n,self.m = self.D.shape
    
    def _Chen_ranking_value(self,A):
        std = lambda x: (sum(array(x - sum(x)/len(x))**2)/len(x))**(1/2)
        return (A.umf_left + A.umf_right)/2 + A.umf_left_apex + A.umf_right_apex + \
            (A.lmf_left + A.lmf_right)/2 + A.lmf_left_apex + A.lmf_right_apex + \
                - 1/4*(std([A.umf_left,A.umf_left_apex]) + std([A.umf_left_apex,A.umf_right_apex]) + std([A.umf_right_apex,A.umf_right]) + std([A.umf_left,A.umf_left_apex,A.umf_right_apex,A.umf_right])) \
                    - 1/4*(std([A.lmf_left,A.lmf_left_apex]) + std([A.lmf_left_apex,A.lmf_right_apex]) + std([A.lmf_right_apex,A.lmf_right]) + std([A.lmf_left,A.lmf_left_apex,A.lmf_right_apex,A.lmf_right])) \
                        + A.umf_left_height + A.umf_right_height + A.lmf_left_height + A.lmf_right_height

    
    def evaluate(self):
        Y = array([[self._Chen_ranking_value(e) for e in row] for row in self.D])
        positive = Y.max(axis=0,keepdims=True)
        negtive = Y.min(axis=0,keepdims=True)
        dp = (((Y-positive)**2).sum(axis=1))**(1/2)
        dn = (((Y-negtive)**2).sum(axis=1))**(1/2)
        self.Closeness = dn/(dp+dn)
        