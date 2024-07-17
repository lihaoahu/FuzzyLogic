from functools import cached_property

from numpy import sqrt, mean, abs
from pandas import DataFrame

from fuzzy_logic.interval_type_2.fuzzy_sets import TrapezoidalIT2FS
from perceptual_computer.encoder import Encoder
from perceptual_computer.encoder.interval_approach import (
    _is_reasonable_interval_ia, _not_bad_data_ia)
from perceptual_computer.encoder.enhanced_interval_approach import _in_tolerance_limit_eia, _not_outlier_eia
from perceptual_computer.encoder.interval_to_t1fs import ApexFixedTrapezoidalModel

def fou_classify_hma(data,M=10):
    n = len(data)
    K = "26.260 26.260 7.656 5.144 4.203 " + \
        "3.708 3.400 3.187 3.031 2.911 " + \
        "2.815 2.735 2.671 2.615 2.566 " + \
        "2.524 2.486 2.453 2.423 2.396 " + \
        "2.396 " * 4 + "2.292 " + \
        "2.292 " * 4 + "2.220 " + \
        "2.220 " * 4 + "2.167 " + \
        "2.167 " * 4 + "2.126 " + \
        "2.126 " * 4 + "2.092 " + \
        "2.092 " * 4 + "2.065 " + \
        "2.065 " * 9 + "2.022 " + \
        "2.022 " * 9 + "1.990 " + \
        "1.990 " * 9 + "1.965 " + \
        "1.965 " * 9 + "1.944 " + \
        "1.944 " * 9 + "1.927 " + \
        "1.927 " * 49 + "1.870 " + \
        "1.870 " * 49 + "1.837 " + \
        "1.837 " * 49 + "1.815 " + \
        "1.815 " * 49 + "1.800"
    K = K.split(" ")
    K = [eval(k) for k in K]
    if n > 300:
        k = 1.645
    else:
        k = K[n-1]
    al = data.left.mean() - k*data.left.std()
    bu = data.right.mean() + k*data.right.std()
    flag1 = al >= 0
    flag2 = bu <= M
    fou_shape = int(flag1) + 2* int(flag2)
    if fou_shape == 0:
        fou_shape = 3
    enum = ["NO FOU","RIGHT-SHOULDER","LEFT-SHOULDER","INTERIOR"]
    return enum[fou_shape]


class HaoMendelApproach(Encoder):
    name = "Hao-Mendel Approach"

    def __init__(self, l, r, M=10):
        self.data = DataFrame(
            {"left": l, "right": r, "length": [i-j for i, j, in zip(r, l)]})
        self.M = M

    def bad_data_processing(self):
        not_bad_data_index = _not_bad_data_ia(self.data, self.M)
        self.data = self.data.iloc[not_bad_data_index]

    def outlier_processing(self):
        _, not_outlier_index = _not_outlier_eia(self.data)
        self.data = self.data.iloc[not_outlier_index]

    def tolerance_limit_processing(self):
        _, in_tolerance_limit_index = _in_tolerance_limit_eia(self.data)
        self.data = self.data.iloc[in_tolerance_limit_index]

    def reasonable_interval_processing(self):
        is_reasonable = _is_reasonable_interval_ia(self.data)
        self.data = self.data.iloc[is_reasonable]

    def data_part(self):
        self.bad_data_processing()
        self.outlier_processing()
        self.tolerance_limit_processing()
        self.reasonable_interval_processing()

    def fou_classify(self):
        self.fou_shape = fou_classify_hma(self.data,self.M)
        intras = {
            "LEFT-SHOULDER" : ApexFixedTrapezoidalModel(apexes=[0,min(self.data.right)]),
            "INTERIOR" : ApexFixedTrapezoidalModel(apexes=[max(self.data.left),min(self.data.right)]),
            "RIGHT-SHOULDER" : ApexFixedTrapezoidalModel(apexes=[max(self.data.left),10]),
        }
        self.intra = intras[self.fou_shape]
        self.data['Embedded FS'] = [self.intra.interval_to_t1fs([row.left, row.right]) for i, row in
                                    self.data.iterrows()]
    
    @cached_property
    def IT2FS(self):
        self.fou_classify()
        self.overlap = [max(self.data.left),min(self.data.right)]
        ol,ou = self.overlap
        
        if self.fou_shape == "INTERIOR":
            # Left
            m_LH = ol - mean(abs(self.data.left - ol))
            s_LH = self.data.left.std()
            al = max([0,ol-3*sqrt(2)*s_LH])
            ar = min([ol,6*m_LH+3*sqrt(2)*s_LH-5*ol])
            # Right
            m_RH = ou + mean(abs(self.data.right-ou))
            s_RH = self.data.right.std()
            br = min(self.M,ou+3*sqrt(2)*s_RH)
            bl = max([ou,6*m_RH-3*sqrt(2)*s_RH-5*ou])
            return TrapezoidalIT2FS(ar,ol,ou,bl,1,1,al,ol,ou,br,1,1)
        elif self.fou_shape == "LEFT-SHOULDER":
            # Right
            m_RH = ou + mean(abs(self.data.right-ou))
            s_RH = self.data.right.std()
            bl = min(self.M,ou+3*sqrt(2)*s_RH)
            br = max([ou,6*m_RH-3*sqrt(2)*s_RH-5*ou])
            return TrapezoidalIT2FS(0,0,ou,bl,1,1,0,0,ou,br,1,1)
        elif self.fou_shape == "RIGHT-SHOULDER":
            # Left
            m_LH = ol - mean(abs(self.data.left - ol))
            s_LH = self.data.left.std()
            al = max([0,ol-3*sqrt(2)*s_LH])
            ar = min([ol,6*m_LH+3*sqrt(2)*s_LH-5*ol])
            return TrapezoidalIT2FS(ar,ol,self.M,self.M,1,1,al,ol,self.M,self.M,1,1)
        elif self.fou_shape == "NO FOU":
            return None