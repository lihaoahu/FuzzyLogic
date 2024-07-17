from functools import cached_property
from math import log, sqrt

from fuzzy_logic.interval_type_2.fuzzy_sets import TrapezoidalIT2FS
from fuzzy_logic.type_1.fuzzy_sets import (TrapezoidalFuzzySet,
                                           TriangularFuzzyNumber)
from perceptual_computer.encoder.interval_to_t1fs import SymmetricTriangularModel,LeftShoulderTrapezoidalModel,RightShoulderTrapezoidalModel

from numpy import quantile
from pandas import DataFrame
from perceptual_computer.encoder import Encoder
from scipy.stats import t


def _not_bad_data_ia(data, M=10):
    index = []
    for i, (left, right) in enumerate(zip(data.left, data.right)):
        if (
            0 <= left <= M and
            0 <= right <= M and
            0 < right-left < M
        ):
            index.append(i)
    return index


def _not_outlier_ia(data, bound=0.25):
    ql25 = quantile(data.left, 0.25)
    ql75 = quantile(data.left, 0.75)
    qr25 = quantile(data.right, 0.25)
    qr75 = quantile(data.right, 0.75)
    qlen25 = quantile(data.length, 0.25)
    qlen75 = quantile(data.length, 0.75)
    lqrl = ql75 - ql25
    lqrr = qr75 - qr25
    lqrlen = qlen75 - qlen25

    index = []
    for i, (left, right, length) in enumerate(zip(data.left, data.right, data.length)):
        if (
            (lqrl < bound or ql25-1.5*lqrl <= left <= ql75+1.5*lqrl) and
            (lqrr < bound or qr25-1.5*lqrr <= right <= qr75+1.5*lqrr) and
            (lqrlen < bound or qlen25-1.5*lqrlen <= length <= qlen75+1.5*lqrlen)
        ):
            index.append(i)
    return index


def _in_tolerance_limit_ia(data, bound=0.25):
    n = len(data)
    data_mean = data.mean()
    data_std = data.std()
    K = '32.019 32.019 8.380 5.369 4.275 3.712 3.369 3.136 2.967 2.839 2.737 2.655 2.587 2.529 2.48 2.437 2.4 2.366 2.337 2.31 2.31 2.31 2.31 2.31 2.208'
    K = K.split(" ")
    K = [eval(k) for k in K]
    k = min(n, 25)
    k = K[k-1]
    index = []
    for i, (left, right, length) in enumerate(zip(data.left, data.right, data.length)):
        if (
            (data_std.left < bound or data_mean.left-k*data_std.left <= left <= data_mean.left+k*data_std.left) and
            (data_std.right < bound or data_mean.right-k*data_std.right <= right <= data_mean.right+k*data_std.right) and
            (data_std.length < bound or data_mean.length-k *
             data_std.length <= length <= data_mean.length+k*data_std.length)
        ):
            index.append(i)
    return index


def _is_reasonable_interval_ia(data):
    data_mean = data.mean()
    data_std = data.std()
    ml, mr = data_mean.left, data_mean.right
    sl, sr = data_std.left, data_std.right
    if sl == sr:
        barrier = (ml+mr)/2
    elif sl == 0:
        barrier = ml + 0.01
    elif sr == 0:
        barrier = mr - 0.01
    else:
        barrier1 = ((mr*sl**2 - ml*sr**2)+sl*sr*((ml-mr)**2 +
                                                 2*(sl**2-sr**2)*log(sl/sr))**0.5)/(sl**2-sr**2)
        barrier2 = ((mr*sl**2 - ml*sr**2)-sl*sr*((ml-mr)**2 +
                                                 2*(sl**2-sr**2)*log(sl/sr))**0.5)/(sl**2-sr**2)
        if barrier1 <= mr and barrier1 >= ml:
            barrier = barrier1
        else:
            barrier = barrier2
    index = []
    for i, (left, right) in enumerate(zip(data.left, data.right)):
        if (
            2*ml-barrier <= left < barrier < right <= 2*mr-barrier
        ):
            index.append(i)
    return index

def fou_classify_ia(data):
    M = data.mean()
    data["C"] = data.right - 5.831 * data.left
    data["D"] = data.right - 0.171 * data.left - 8.29
    S = data.std()
    n = len(data)
    talpha = t.ppf(0.95, n - 1)
    shift1 = talpha * S.C / sqrt(n)
    shift2 = talpha * S.D / sqrt(n)
    flag1 = M.right <= 5.831 * M.left - shift1
    flag2 = M.right <= 8.29 + 0.171 * M.left - shift2
    fou_shape = int(flag1) + 2 * int(flag2)
    if fou_shape == 0:
        fou_shape = 3
    enum = ["NO FOU", "RIGHT-SHOULDER",
            "LEFT-SHOULDER", "INTERIOR"]
    return enum[fou_shape]


class IntervalApproach(Encoder):
    name = "Interval Approach"

    def __init__(self, l, r, M=10):
        self.data = DataFrame(
            {"left": l, "right": r, "length": [i - j for i, j, in zip(r, l)]})
        self.M = M
        self.intra = None
        self.intras = {
            "INTERIOR" : SymmetricTriangularModel(),
            "LEFT-SHOULDER" : LeftShoulderTrapezoidalModel(),
            "RIGHT-SHOULDER" : RightShoulderTrapezoidalModel(M=self.M),
            "NO FOU" : None
        }

    def bad_data_processing(self):
        not_bad_data_index = _not_bad_data_ia(self.data, self.M)
        self.data = self.data.iloc[not_bad_data_index]

    def outlier_processing(self):
        not_outlier_index = _not_outlier_ia(self.data)
        self.data = self.data.iloc[not_outlier_index]

    def tolerance_limit_processing(self):
        in_tolerance_limit_index = _in_tolerance_limit_ia(self.data)
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
        return fou_classify_ia(self.data)

    def compute_et1fs(self):
        self.data['Embedded FS'] = [self.intra.interval_to_t1fs([row.left, row.right]) for i, row in
                                    self.data.iterrows()]
        if self.fou_shape == "INTERIOR":
            self.data["amf"] = [t1fs.left for t1fs in self.data['Embedded FS']]
            self.data["bmf"] = [t1fs.right for t1fs in self.data['Embedded FS']]
        elif self.fou_shape == "LEFT-SHOULDER":
            self.data["amf"] = [t1fs.mid_right for t1fs in self.data['Embedded FS']]
            self.data["bmf"] = [t1fs.right for t1fs in self.data['Embedded FS']]
        elif self.fou_shape == "RIGHT-SHOULDER":
            self.data["amf"] = [t1fs.left for t1fs in self.data['Embedded FS']]
            self.data["bmf"] = [t1fs.mid_left for t1fs in self.data['Embedded FS']]
        elif self.fou_shape == "NO FOU":
            self.data["amf"] = 0
            self.data["bmf"] = self.M

        index = [a >= 0 and b <= self.M for a,
                 b in zip(self.data.amf, self.data.bmf)]
        self.data = self.data[index]

    @cached_property
    def IT2FS(self):
        self.fou_shape = self.fou_classify()
        self.intra = self.intras[self.fou_shape]
        self.compute_et1fs()
        self.data["cmf"] = (self.data.amf + self.data.bmf)/2
        am, aM = self.data.amf.min(), self.data.amf.max()
        bm, bM = self.data.bmf.min(), self.data.bmf.max()
        cm, cM = self.data.cmf.min(), self.data.cmf.max()
        if self.fou_shape == "INTERIOR":
            p = (bm*(cM-aM)+aM*(bm-cm))/((cM-aM)+(bm-cm))
            mu_p = (bm-p)/(bm-cm)
            return TrapezoidalIT2FS(am, cm, cM, bM, 1, 1, aM, p, p, bm, mu_p, mu_p)
        elif self.fou_shape == "LEFT-SHOULDER":
            return TrapezoidalIT2FS(0, 0, aM, bM, 1, 1, 0, 0, am, bm, 1, 1)
        elif self.fou_shape == "RIGHT-SHOULDER":
            return TrapezoidalIT2FS(am, bm, self.M, self.M, 1, 1, aM, bM, self.M, self.M, 1, 1)
        elif self.fou_shape == "NO FOU":
            return None
