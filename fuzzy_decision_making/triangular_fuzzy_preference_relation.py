from numpy import array, zeros, round, argmax
from functools import cached_property
from numpy.linalg import eig
import geatpy as ea

from fuzzy_logic.type_1.fuzzy_sets import TriangularFuzzyNumber as TFN


RI = {
    3: 0.52,
    4: 0.89,
    5: 1.12,
    6: 1.26,
    7: 1.36,
    8: 1.41,
    9: 1.46,
    10: 1.49,
    11: 1.52,
    12: 1.54
}


class MultiplicativeTriangularFuzzyPreferenceRelation:
    def __init__(self, mat) -> None:
        self.mat = array(mat)
        n, m = self.mat.shape
        assert n == m
        self.n = n

    def lambda_l(self):
        indices = [(i, j) for i in range(self.n)
                   for j in range(self.n) if i < j]

        def evalVars(Vars):
            n_pop, _ = Vars.shape
            ObjV = []
            for k in range(n_pop):
                P = zeros((self.n, self.n))
                for i in range(self.n):
                    P[i, i] = 1
                for i, index in enumerate(indices):
                    P[index] = Vars[k, i]
                    P[index[1], index[0]] = 1/Vars[k, i]
                lamb, w = eig(P)
                index = argmax(lamb)
                lamb, w = lamb[index].real, w[:, index].real
                ObjV.append([lamb])
            return array(ObjV)

        problem = ea.Problem(
            name='',
            M=1,
            maxormins=[1],
            Dim=len(indices),
            varTypes=[0] * len(indices),
            lb=[self.mat[index].left for index in indices],
            ub=[self.mat[index].right for index in indices],
            evalVars=evalVars
        )

        algorithm = ea.soea_SEGA_templet(
            problem,
            ea.Population(Encoding='RI', NIND=20),
            MAXGEN=50,
            logTras=10,
            trappedValue=1e-6,
            maxTrappedCount=10
        )

        res = ea.optimize(algorithm, verbose=False, drawing=0,
                          outputMsg=False, drawLog=False, saveFlag=False)
        P = zeros((self.n, self.n))
        for i in range(self.n):
            P[i, i] = 1
        for i, index in enumerate(indices):
            P[index] = res['Vars'][0, i]
            P[index[1], index[0]] = 1/P[index]
        lamb, w = eig(P)
        index = argmax(lamb)
        lamb, w = lamb[index].real, w[:, index].real
        w = abs(w)
        w = w/w.sum()
        return lamb, w, P

    def lambda_m(self):
        M = array([[self.mat[i, j].middle for j in range(self.n)]
                   for i in range(self.n)])
        lamb, w = eig(M)
        index = argmax(lamb)
        lamb, w = lamb[index].real, w[:, index].real
        w = abs(w)
        w = w/w.sum()
        return lamb, w, M

    def lambda_u(self):
        indices = [(i, j) for i in range(self.n)
                   for j in range(self.n) if i < j]

        def evalVars(Vars):
            n_pop, _ = Vars.shape
            ObjV = []
            for k in range(n_pop):
                P = zeros((self.n, self.n))
                for i in range(self.n):
                    P[i, i] = 1
                for i, index in enumerate(indices):
                    P[index] = Vars[k, i]
                    P[index[1], index[0]] = 1/Vars[k, i]
                lamb, w = eig(P)
                index = argmax(lamb)
                lamb, w = lamb[index].real, w[:, index].real
                ObjV.append([lamb])
            return array(ObjV)

        problem = ea.Problem(
            name='',
            M=1,
            maxormins=[-1],
            Dim=len(indices),
            varTypes=[0] * len(indices),
            lb=[self.mat[index].left for index in indices],
            ub=[self.mat[index].right for index in indices],
            evalVars=evalVars
        )

        algorithm = ea.soea_SEGA_templet(
            problem,
            ea.Population(Encoding='RI', NIND=20),
            MAXGEN=50,
            logTras=10,
            trappedValue=1e-6,
            maxTrappedCount=10
        )

        res = ea.optimize(algorithm, verbose=False, drawing=0,
                          outputMsg=False, drawLog=False, saveFlag=False)
        P = zeros((self.n, self.n))
        for i in range(self.n):
            P[i, i] = 1
        for i, index in enumerate(indices):
            P[index] = res['Vars'][0, i]
            P[index[1], index[0]] = 1/P[index]
        lamb, w = eig(P)
        index = argmax(lamb)
        lamb, w = lamb[index].real, w[:, index].real
        w = abs(w)
        w = w/w.sum()
        return lamb, w, P

    @cached_property
    def FCI(self):
        n = self.n
        lamb_l, _, _ = self.lambda_l()
        lamb_u, _, _ = self.lambda_u()
        lamb_m, _, _ = self.lambda_m()
        return TFN(round((lamb_l-n)/(n-1), 4), round((lamb_m-n)/(n-1), 4), round((lamb_u-n)/(n-1), 4))

    @cached_property
    def FCR(self):
        return TFN(self.FCI.left / RI[self.n], self.FCI.middle / RI[self.n], self.FCI.right / RI[self.n])

    def alpha(self, I):
        indices = [(i, j) for i in range(self.n)
                   for j in range(self.n) if i < j]

        def evalVars(Vars):
            n_pop, _ = Vars.shape
            ObjV = []
            for k in range(n_pop):
                P = zeros((self.n, self.n))
                for i in range(self.n):
                    P[i, i] = 1
                for i, index in enumerate(indices):
                    P[index] = Vars[k, i]
                    P[index[1], index[0]] = 1/Vars[k, i]
                lamb, w = eig(P)
                index = argmax(lamb)
                lamb, w = lamb[index].real, w[:, index].real
                w = abs(w)
                w = w/w.sum()
                ObjV.append([w[I]])
            return array(ObjV)

        problem = ea.Problem(
            name='',
            M=1,
            maxormins=[1],
            Dim=len(indices),
            varTypes=[0] * len(indices),
            lb=[self.mat[index].left for index in indices],
            ub=[self.mat[index].right for index in indices],
            evalVars=evalVars
        )

        algorithm = ea.soea_SEGA_templet(
            problem,
            ea.Population(Encoding='RI', NIND=20),
            MAXGEN=50,
            logTras=10,
            trappedValue=1e-6,
            maxTrappedCount=10
        )

        res = ea.optimize(algorithm, verbose=False, drawing=0,
                          outputMsg=False, drawLog=False, saveFlag=False)
        P = zeros((self.n, self.n))
        for i in range(self.n):
            P[i, i] = 1
        for i, index in enumerate(indices):
            P[index] = res['Vars'][0, i]
            P[index[1], index[0]] = 1/P[index]
        lamb, w = eig(P)
        index = argmax(lamb)
        lamb, w = lamb[index].real, w[:, index].real
        w = abs(w)
        w = w/w.sum()
        return w[I]

    def beta(self, I):
        _, w_m, _ = self.lambda_m()
        return w_m[I]

    def gamma(self, I):
        indices = [(i, j) for i in range(self.n)
                   for j in range(self.n) if i < j]

        def evalVars(Vars):
            n_pop, _ = Vars.shape
            ObjV = []
            for k in range(n_pop):
                P = zeros((self.n, self.n))
                for i in range(self.n):
                    P[i, i] = 1
                for i, index in enumerate(indices):
                    P[index] = Vars[k, i]
                    P[index[1], index[0]] = 1/Vars[k, i]
                lamb, w = eig(P)
                index = argmax(lamb)
                lamb, w = lamb[index].real, w[:, index].real
                w = abs(w)
                w = w/w.sum()
                ObjV.append([w[I]])
            return array(ObjV)

        problem = ea.Problem(
            name='',
            M=1,
            maxormins=[-1],
            Dim=len(indices),
            varTypes=[0] * len(indices),
            lb=[self.mat[index].left for index in indices],
            ub=[self.mat[index].right for index in indices],
            evalVars=evalVars
        )

        algorithm = ea.soea_SEGA_templet(
            problem,
            ea.Population(Encoding='RI', NIND=20),
            MAXGEN=50,
            logTras=10,
            trappedValue=1e-6,
            maxTrappedCount=10
        )

        res = ea.optimize(algorithm, verbose=False, drawing=0,
                          outputMsg=False, drawLog=False, saveFlag=False)
        P = zeros((self.n, self.n))
        for i in range(self.n):
            P[i, i] = 1
        for i, index in enumerate(indices):
            P[index] = res['Vars'][0, i]
            P[index[1], index[0]] = 1/P[index]
        lamb, w = eig(P)
        index = argmax(lamb)
        lamb, w = lamb[index].real, w[:, index].real
        w = abs(w)
        w = w/w.sum()
        return w[I]

    def prior(self):
        _, beta, _ = self.lambda_m()
        return tuple(TFN(round(self.alpha(i), 4), round(beta[i], 4), round(self.gamma(i), 4)) for i in range(self.n))

    def __repr__(self) -> str:
        return self.mat.__repr__()

    def __str__(self) -> str:
        return self.__repr__()
