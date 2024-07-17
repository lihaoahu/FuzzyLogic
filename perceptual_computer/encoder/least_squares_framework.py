import geatpy as ea
from fuzzy_logic.interval_type_2.fuzzy_sets import TrapezoidalIT2FS
from numba import jit, prange
from numpy import (argsort, array, float64, linspace, mean, quantile, repeat,
                   sqrt, zeros,min)
from numpy.random import normal
from scipy.optimize import minimize
from pandas import DataFrame
from perceptual_computer.encoder import Encoder
from perceptual_computer.encoder.enhanced_interval_approach import (
    _in_tolerance_limit_eia, _not_outlier_eia)
from perceptual_computer.encoder.interval_approach import (
    _is_reasonable_interval_ia, _not_bad_data_ia)
from perceptual_computer.encoder.cy_regression_approach import cy_fuzzy_statistic


def fuzzy_statistic(data, N_bins=100,begin=0.,end=10.):
    left = data.left.to_numpy()
    right = data.right.to_numpy()
    return cy_fuzzy_statistic(left, right, N_bins, begin,end)

def interrelated_degree(mu1, mu2):
    return 2*sum(min([mu1, mu2], axis=0))/(sum(mu1)+sum(mu2))


def compatibility(data):
    x, mu_x = fuzzy_statistic(data)
    similarity = []
    for t1fs in data["Embedded FS"]:
        efs_mu_x = t1fs(x)
        similarity.append(
            sum(efs_mu_x*mu_x)/sqrt(sum(efs_mu_x**2)*sum(mu_x**2))
        )
    return mean(similarity)


def compatibility_riemann(efss,x,mu_x):
    similarity = []
    for t1fs in efss:
        efs_mu_x = t1fs(x)
        similarity.append(
            interrelated_degree(mu_x, efs_mu_x)
        )
    return mean(similarity)

@jit(nopython=True, parallel=True)
def errors_numba(t1fs, x, mf_observed):
    a,b,c,d,h = t1fs
    if not a<=b<=c<=d:
        return 1000
    h1 = h2 = h
    b_a = b-a
    c_b = c-b
    h2_h1 = h2-h1
    c_d = c-d
    sum_of_error = 0.0
    for i in prange(len(x)):
        xi = x[i]
        mf_observed_i = mf_observed[i]
        if a <= xi <= b:
            sum_of_error += (mf_observed_i - (xi-a)/b_a*h1)**2
        elif b < xi <= c:
            sum_of_error += (mf_observed_i - h1 - (xi-b)/c_b*h2_h1)**2
        elif c < xi <= d:
            sum_of_error += (mf_observed_i - h2*(xi-d)/c_d)**2
        else:
            sum_of_error += mf_observed_i**2
    return sum_of_error


@jit(nopython=True, parallel=True)
def errors_numba_vector(A, B, C, D, H1, H2, X, MF,NIND,N):
    errors = zeros((NIND, 1), dtype=float64)
    for i in prange(NIND):
        if A[i] <= B[i] <= C[i] <= D[i]:
            a = A[i]
            b = B[i]
            c = C[i]
            d = D[i]
            h1 = H1[i]
            h2 = H2[i]
            b_a = b-a
            c_b = c-b
            h2_h1 = h2-h1
            c_d = c-d
            sum_of_error = 0.0
            for j in range(N):
                xj = X[j]
                mfj = MF[j]
                if a <= xj <= b:
                    sum_of_error += (mfj - (xj-a)/b_a*h1)**2
                elif b < xj <= c:
                    sum_of_error += (mfj - h1 - (xj-b)/c_b*h2_h1)**2
                elif c < xj <= d:
                    sum_of_error += (mfj - h2*(xj-d)/c_d)**2
                else:
                    sum_of_error += mfj**2
            errors[i, 0] = sum_of_error
        else:
            errors[i, 0] = 1000.0
    return errors


class TrapezoidalIT2FS_constructor(ea.Problem):  # 继承Problem父类
    def __init__(self, x, lmf, umf):
        name = 'Trapezoidal FOU Fitting'  # 初始化name（函数名称，可以随意设置

        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 10  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        bounds = ([[x.min(), x.max()]]*4 + [[0, 1]]) * 2
        lb = [bound[0] for bound in bounds]  # 决策变量下界
        ub = [bound[1] for bound in bounds]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        self.x = x
        self.lmf = lmf
        self.umf = umf
        self.x_start = x.min()
        self.x_end = x.max()
        self.NIND = 50
        self.flag = 0
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        if self.flag == 0:
            x = self.x
            alpha = 0.7
            x_start = self.x_start
            x_end = self.x_end
            pop.Phen = repeat([[x_start, x_start, x_end, x_end, 1, alpha*x_start+(1-alpha)*x_end, alpha*x_start+(1-alpha)*x_end, alpha*x_end+(
                1-alpha)*x_start, alpha*x_end+(1-alpha)*x_start, 1]], self.NIND, axis=0) + 0.5*(x[-1]-x[0])*normal(0, 1, [self.NIND, 10])
            # print(x_start, x_start, x_end, x_end, 1, alpha*x_start+(1-alpha)*x_end, alpha*x_start +
            #       (1-alpha)*x_end, alpha*x_end+(1-alpha)*x_start, alpha*x_end+(1-alpha)*x_start, 1)
            self.flag += 1
        Phen = pop.Phen  # 得到决策变量矩阵
        umf_left = Phen[:, [0]]
        umf_left_apex = Phen[:, [1]]
        umf_right_apex = Phen[:, [2]]
        umf_right = Phen[:, [3]]
        umf_right_height = umf_left_height = Phen[:, [4]]

        lmf_left = Phen[:, [5]]
        lmf_left_apex = Phen[:, [6]]
        lmf_right_apex = Phen[:, [7]]
        lmf_right = Phen[:, [8]]
        lmf_right_height = lmf_left_height = Phen[:, [9]]

        pop.ObjV = errors_numba_vector(
            umf_left[:, 0],
            umf_left_apex[:, 0],
            umf_right_apex[:, 0],
            umf_right[:, 0],
            umf_left_height[:, 0],
            umf_right_height[:, 0],
            self.x, self.umf,self.NIND,len(self.x)
        ) + errors_numba_vector(
            lmf_left[:, 0],
            lmf_left_apex[:, 0],
            lmf_right_apex[:, 0],
            lmf_right[:, 0],
            lmf_left_height[:, 0],
            lmf_right_height[:, 0],
            self.x, self.lmf,self.NIND,len(self.x)
        )

    def solve(self):
        problem = self
        """==================================种群设置=============================="""
        Encoding = 'RI'  # 编码方式
        NIND = self.NIND  # 种群规模
        Field = ea.crtfld(Encoding, problem.varTypes,
                          problem.ranges, problem.borders)  # 创建区域描述器
        # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
        population = ea.Population(Encoding, Field, NIND)
        """================================算法参数设置============================="""
        myAlgorithm = ea.soea_DE_rand_1_L_templet(
            problem, population)  # 实例化一个算法模板对象
        myAlgorithm.MAXGEN = 10000  # 最大进化代数
        # myAlgorithm.logTras = 100  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
        myAlgorithm.verbose = False  # 设置是否打印输出日志信息
        myAlgorithm.mutOper.F = 0.5  # 差分进化中的参数F
        myAlgorithm.recOper.XOVR = 0.5  # 重组概率
        myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
        myAlgorithm.trappedValue=1e-8
        myAlgorithm.maxTrappedCount=1000
        # prophetChrom = np.repeat([[1,3,4,6,1,1,2,3,4,5,1,1]],NIND,axis=0)
        # prophetPop = ea.Population(Encoding,Field,NIND,prophetChrom)
        # myAlgorithm.call_aimFunc(prophetPop)
        """===========================调用算法模板进行种群进化========================"""
        [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
        """==================================输出结果=============================="""
        # print('评价次数：%s' % myAlgorithm.evalsNum)
        print('时间已过 %s 秒' % myAlgorithm.passTime)
        if BestIndi.sizes != 0:
            # print('最优的目标函数值为：%s' % BestIndi.ObjV[0][0])
            # if BestIndi.ObjV[0][0] > 10
            # print('最优的控制变量值为：')
            p = BestIndi.Phen[0]
            it2fs = TrapezoidalIT2FS(
                p[0], p[1], p[2], p[3], p[4], p[4], p[5], p[6], p[7], p[8], p[9], p[9])
            # print(it2fs)
            return it2fs,myAlgorithm
        else:
            print('没找到可行解。')
            return None

# class TrapezoidalIT2FS_constructor:
#     def __init__(self,x,lmf,umf) -> None:
#         self.x = x
#         self.lmf = lmf
#         self.umf = umf
    
#     def solve(self):
#         def object_fun(it2fs,x,lmf,umf):
#             err = errors_numba(it2fs[:5],x,umf) + errors_numba(it2fs[5:],x,lmf)
#             return err
#         x,lmf,umf = self.x,self.lmf,self.umf
#         bnds = [[0,x.max()+1e-10]] * 4 + [[0,1]] + [[0,x.max()+1e-10]] * 4 + [[0,1]]
#         x0 = [x[umf>0].min(),x[umf>0].mean(),x[umf<1].mean(),x[umf<1].max(),0.99,x[lmf>0].min(),x[lmf>0].mean(),x[lmf<1].mean(),x[lmf<1].max(),0.01]
#         cons = [
#                     {"type":"ineq","fun":lambda x: x[1] - x[0]},
#                     {"type":"ineq","fun":lambda x: x[2] - x[1]},
#                     {"type":"ineq","fun":lambda x: x[3] - x[2]},
#                     {"type":"ineq","fun":lambda x: x[4] - x[9]},
#                     {"type":"ineq","fun":lambda x: x[6] - x[5]},
#                     {"type":"ineq","fun":lambda x: x[7] - x[6]},
#                     {"type":"ineq","fun":lambda x: x[8] - x[7]},
#                     {"type":"ineq","fun":lambda x: x[5] - x[0]},
#                     {"type":"ineq","fun":lambda x: x[6] - x[1]},
#                     {"type":"ineq","fun":lambda x: x[2] - x[7]},
#                     {"type":"ineq","fun":lambda x: x[3] - x[8]}
#                 ]
#         res = minimize(object_fun,x0=x0,args=(x,lmf,umf),method="SLSQP",bounds=bnds,constraints=cons,tol=1e-5,options={"maxiter":10000,'eps': 1/100})
#         if res.success:
#             it2fs = TrapezoidalIT2FS(
#                 res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[4],
#                 res.x[5],res.x[6],res.x[7],res.x[8],res.x[9],res.x[9]
#             )
#             return it2fs
#         else:
#             print(res.message)
#             return None


class LeastSquaresFramework(Encoder):
    name = 'Regression Approach'

    def __init__(self, l, r, M=10):
        self.intra = None
        self.data = DataFrame(
            {"left": l, "right": r, "length": [i - j for i, j, in zip(r, l)]})
        self.M = M
        self.log = dict()

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

    def _set_intrapersonal_model(self, model):
        self.intra = model
        self.data['Embedded FS'] = [self.intra.interval_to_t1fs([row.left, row.right]) for i, row in
                                    self.data.iterrows()]

    def intrapersonal_model_select(self,models):
        x,mu_x = fuzzy_statistic(self.data)
        compatibilities = []
        for model in models:
            self._set_intrapersonal_model(model)
            compatibilities.append(compatibility_riemann(self.data["Embedded FS"],x,mu_x))
        best_model = models[argsort(compatibilities)[-1]]
        self._set_intrapersonal_model(best_model)
        return {models[i]: compatibilities[i] for i in range(len(models))}

    def natural_fou(self, alpha=0.1, reduction=True):
        data = []
        N = 1000
        x = linspace(0, self.M, N)
        for i in x:
            samples = [t1fs(i) for t1fs in self.data['Embedded FS']]
            data.append(samples)
        natural_lmf, natural_umf = array([quantile(samples, alpha/2) for samples in data]), array(
            [quantile(samples, 1-alpha/2) for samples in data])
        if reduction:
            data = []
            N = 100
            index = natural_umf > 0
            x_start = x[index].min()
            x_end = x[index].max()
            L = x_end - x_start
            x_start -= 0.1*L
            x_end += 0.1*L
            x_start = max([x_start, 0])
            x_end = min([x_end, self.M])
            x = linspace(x_start, x_end, N)
            for i in x:
                samples = [t1fs(i) for t1fs in self.data['Embedded FS']]
                data.append(samples)
            natural_lmf, natural_umf = array([quantile(samples, alpha/2) for samples in data]), array(
                [quantile(samples, 1-alpha/2) for samples in data])
            return x, natural_lmf, natural_umf
        else:
            return x, natural_lmf, natural_umf
        
        
    def IT2FS(self, alpha=0.1):
        x, lmf, umf = self.natural_fou(alpha=alpha)
        constructor = TrapezoidalIT2FS_constructor(x,lmf,umf)
        it2fs,self.log[alpha] = constructor.solve()
        return it2fs
