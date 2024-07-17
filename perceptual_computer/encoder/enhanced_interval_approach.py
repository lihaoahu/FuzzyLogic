from functools import cached_property
from itertools import combinations

from fuzzy_logic.interval_type_2.fuzzy_sets import TrapezoidalIT2FS
from numpy import quantile
from numpy.random import randint
from pandas import DataFrame
from perceptual_computer.encoder import Encoder
from perceptual_computer.encoder.interval_approach import (
    _is_reasonable_interval_ia, _not_bad_data_ia, fou_classify_ia)
from perceptual_computer.encoder.interval_to_t1fs import (
    LeftShoulderTrapezoidalModel, RightShoulderTrapezoidalModel,
    SymmetricTriangularModel)


def _not_outlier_eia(data, bound=0.25):
    ql25 = quantile(data.left, 0.25)
    ql75 = quantile(data.left, 0.75)
    qr25 = quantile(data.right, 0.25)
    qr75 = quantile(data.right, 0.75)
    lqrl = ql75 - ql25
    lqrr = qr75 - qr25
    index = []
    for i, (left, right) in enumerate(zip(data.left, data.right)):
        if (
            (lqrl < bound or ql25-1.5*lqrl <= left <= ql75+1.5*lqrl) and
            (lqrr < bound or qr25-1.5*lqrr <= right <= qr75+1.5*lqrr)
        ):
            index.append(i)

    data1 = data.iloc[index]
    qlen25 = quantile(data1.length, 0.25)
    qlen75 = quantile(data1.length, 0.75)
    lqrlen = qlen75 - qlen25
    index2 = []
    for i, length in zip(index, data1.length):
        if (
            i in index and
            (lqrlen < bound or qlen25-1.5*lqrlen <= length <= qlen75+1.5*lqrlen)
        ):
            index2.append(i)
    return index, index2


def _in_tolerance_limit_eia(data, bound=0.25):
    n = len(data)
    NN = 2000
    AA = randint(n, size=(n*NN))
    resample_data = data.iloc[AA, :]
    resample_data_mean = resample_data.mean()
    resample_data_std = resample_data.std()
    K = '32.019 32.019 8.380 5.369 4.275 3.712 3.369 3.136 2.967 2.839 2.737 2.655 2.587 2.529 2.48 2.437 2.4 2.366 2.337 2.31 2.31 2.31 2.31 2.31 2.208'
    K = K.split(" ")
    K = [eval(k) for k in K]
    k = min(n, 25)
    k = K[k-1]
    index = []
    for i, (left, right) in enumerate(zip(data.left, data.right)):
        if (
            (resample_data_std.left < bound or resample_data_mean.left-k*resample_data_std.left <= left <= resample_data_mean.left+k*resample_data_std.left) and
            (resample_data_std.right < bound or resample_data_mean.right-k *
             resample_data_std.right <= right <= resample_data_mean.right+k*resample_data_std.right)
        ):
            index.append(i)
    data1 = data.iloc[index]
    n = len(data1)
    AA = randint(n, size=(n*NN))
    resample_data = data1.iloc[AA, :]
    resample_data_mean = resample_data.mean()
    resample_data_std = resample_data.std()
    k = min(n, 25)
    # k = K[k-1]
    k = min(K[k-1], resample_data_mean.length/resample_data_std.length,
            (10-resample_data_mean.length)/resample_data_std.length)
    index2 = []
    for i in index:
        if (
            (resample_data_std.length < bound or resample_data_mean.length-k *
             resample_data_std.length <= data.iloc[i].length <= resample_data_mean.length+k*resample_data_std.length)
        ):
            index2.append(i)
    return index, index2



class EnhancedIntervalApproach(Encoder):
    name = "Enhanced Interval Approach"

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

    def get_p(self):
        max_p = 10
        mu_p = None
        for fs1, fs2 in combinations(self.data['Embedded FS'], 2):
            if fs1.middle > fs2.middle:
                fs1, fs2 = fs2, fs1
            p = (fs1.right - fs2.left) / \
                (fs1.right - fs2.left + fs2.middle - fs1.middle)
            if p < max_p:
                max_p = p
                mu_p = fs2.left + p * (fs2.middle - fs2.left)
        return max_p, mu_p

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
            p, mu_p = self.get_p()
            return TrapezoidalIT2FS(am, cm, cM, bM, 1, 1, aM, mu_p, mu_p, bm, p, p)
        elif self.fou_shape == "LEFT-SHOULDER":
            return TrapezoidalIT2FS(0, 0, aM, bM, 1, 1, 0, 0, am, bm, 1, 1)
        elif self.fou_shape == "RIGHT-SHOULDER":
            return TrapezoidalIT2FS(am, bm, self.M, self.M, 1, 1, aM, bM, self.M, self.M, 1, 1)
        elif self.fou_shape == "NO FOU":
            return None
