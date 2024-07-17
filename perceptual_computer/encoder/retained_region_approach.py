from functools import cached_property
from math import sqrt

import geatpy as ea
from numba import jit,prange
from numpy import all, array, hstack, linspace, quantile, std,abs,ones_like,mean,argsort,isnan,repeat,zeros,float64,ones,empty,nan_to_num,nan
from numpy.random import normal
from pandas import DataFrame
from perceptual_computer.encoder import Encoder
from perceptual_computer.encoder.enhanced_interval_approach import \
    _not_outlier_eia
from perceptual_computer.encoder.least_squares_framework import TrapezoidalIT2FS_constructor
from perceptual_computer.encoder.interval_approach import (
    _is_reasonable_interval_ia, _not_bad_data_ia)
from perceptual_computer.encoder.enhanced_interval_approach import _in_tolerance_limit_eia
from fuzzy_logic.interval_type_2.fuzzy_sets import TrapezoidalIT2FS
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

SQRT_6 = sqrt(6)


@jit(nopython=True)
def mf_trap(x, a, b, c, d, h):
    if x < a:
        return 0
    elif x == a:
        if a == b:
            return h
        else:
            return 0
    elif x <= b:
        return (x-a)/(b-a)*h
    elif x <= c:
        return h
    elif x <= d:
        return (d-x)/(d-c)*h
    else:
        return 0


def mu_trap_x(x, model):
    # @jit(nopython=True)
    def mu(m, s):
        a = model.left_ms(m, s)
        b = model.mid_left_ms(m, s)
        c = model.mid_right_ms(m, s)
        d = model.right_ms(m, s)

        return mf_trap(x, a, b, c, d, 1)
    return mu



def trap_constraints(intra):
    def con(retained_region,m,s):
        m_l, m_u, coef, intercept, residuals_std, ks = retained_region
        a = intra.left_ms(m,s)
        b = intra.mid_left_ms(m,s)
        c = intra.mid_right_ms(m,s)
        d= intra.right_ms(m,s)
        # assert len(a) == len(b) == len(c) == len(d) == len(m) == len(s)
        return [
            -a,
            d - 10,
            a - b,
            b - c,
            c - d
        ]
    return con

def unbounded_trap_constraints(intra):
    def con(retained_region,m,s):
        m_l, m_u, coef, intercept, residuals_std, ks = retained_region
        a = intra.left_ms(m,s)
        b = intra.mid_left_ms(m,s)
        c = intra.mid_right_ms(m,s)
        d = intra.right_ms(m,s)
        # assert len(a) == len(b) == len(c) == len(d) == len(m) == len(s)
        return [
            a - b,
            b - c,
            c - d
        ]
    return con

@jit(nopython=True, parallel=True)
def cv_numba(A, B, C, D, H, X, MF,NIND,N):
    CV = zeros((NIND, N), dtype=float64)
    for i in prange(NIND):
        if A[i] <= B[i] <= C[i] <= D[i]:
            a = A[i]
            b = B[i]
            c = C[i]
            d = D[i]
            h = H[i]
            b_a = b-a
            c_b = c-b
            c_d = c-d
            cv = zeros((N,), dtype=float64)
            for j in range(N):
                xj = X[j]
                mfj = MF[j]
                if a <= xj <= b:
                    cv[j] = mfj - (xj-a)/b_a*h
                elif b < xj <= c:
                    cv[j] = mfj - h
                elif c < xj <= d:
                    cv[j] = mfj - h*(xj-d)/c_d
                else:
                    cv[j] = mfj
            CV[i,:] = cv
        else:
            CV[i,:] = ones((N,), dtype=float64) * nan
    return CV


class RRA_TrapezoidalIT2FS_constructor(ea.Problem):  # 继承Problem父类
    def __init__(self, x, lmf, umf):
        name = 'Trapezoidal FOU Fitting'  # 初始化name（函数名称，可以随意设置
        self.x = x
        self.lmf = lmf
        self.umf = umf
        self.x_start = x.min()
        self.x_end = x.max()
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 10  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        bounds = ([[self.x_start, self.x_end]]*4 + [[0, 1]]) * 2
        lb = [bound[0] for bound in bounds]  # 决策变量下界
        ub = [bound[1] for bound in bounds]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        
        self.NIND = 100
        self.flag = 0
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        # if self.flag == 0:
        if self.flag % 1000 == 0:
            # print(pop.Phen.mean(axis=0))
            pass
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

        
        lt_umf = cv_numba(
            umf_left[:, 0],
            umf_left_apex[:, 0],
            umf_right_apex[:, 0],
            umf_right[:, 0],
            umf_left_height[:, 0],
            self.x, self.umf,self.NIND,len(self.x)
        )
        # gt_umf[gt_umf<0] = 10
        lt_lmf = cv_numba(
            lmf_left[:, 0],
            lmf_left_apex[:, 0],
            lmf_right_apex[:, 0],
            lmf_right[:, 0],
            lmf_left_height[:, 0],
            self.x, self.lmf,self.NIND,len(self.x)
        )
        # lt_lmf[lt_lmf<0] = 10
        # pop.ObjV = (umf_right_apex - umf_left_apex + umf_right- umf_left)*umf_left_height/2 + \
        #     (lmf_right_apex - lmf_left_apex + lmf_right- lmf_left)*lmf_left_height/2 *\
        #         (gt_umf.sum(axis=1,keepdims=True) + lt_lmf.sum(axis=1,keepdims=True))
        # pop.ObjV = hstack([gt_umf.sum(axis=1,keepdims=True) , lt_lmf.sum(axis=1,keepdims=True)])
        # pop.ObjV = abs(lt_umf).sum(axis=1,keepdims=True) + abs(lt_lmf).sum(axis=1,keepdims=True)
        pop.ObjV =(umf_right_apex - umf_left_apex + umf_right- umf_left)*umf_left_height/2 + \
            (lmf_right_apex - lmf_left_apex + lmf_right- lmf_left)*lmf_left_height/2
        pop.ObjV = nan_to_num(pop.ObjV,nan=10000)
        pop.ObjV = pop.ObjV*((lt_umf>0).sum(axis=1,keepdims=True)+(lt_lmf<0).sum(axis=1,keepdims=True))
        pop.CV = hstack([(lt_umf>0).sum(axis=1,keepdims=True),(lt_lmf<0).sum(axis=1,keepdims=True)])

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
        myAlgorithm = ea.soea_EGA_templet(
            problem, population)  # 实例化一个算法模板对象
        myAlgorithm.MAXGEN = 10000 # 最大进化代数
        myAlgorithm.logTras = 1000  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
        myAlgorithm.verbose = True  # 设置是否打印输出日志信息
        # myAlgorithm.mutOper.F = 0.6  # 差分进化中的参数F
        # myAlgorithm.recOper.XOVR = 0.6  # 重组概率
        myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
        myAlgorithm.trappedValue=1e-10
        myAlgorithm.maxTrappedCount=1000
        x = self.x
        alpha = 0.7
        x_start = self.x_start
        x_end = self.x_end
        prophetChrom = array([[x_start, x_start, x_end, x_end, 1, alpha*x_start+(1-alpha)*x_end, alpha*x_start+(1-alpha)*x_end, alpha*x_end+(
            1-alpha)*x_start, alpha*x_end+(1-alpha)*x_start, 1]])
        # prophetChrom = array([[0,3,7,10,1,alpha*x_start+(1-alpha)*x_end,alpha*x_start+(1-alpha)*x_end,alpha*x_start+(1-alpha)*x_end,alpha*x_start+(1-alpha)*x_end,0]])
        # print(x_start, x_start, x_end, x_end, 1, alpha*x_start+(1-alpha)*x_end, alpha*x_start +
        #         (1-alpha)*x_end, alpha*x_end+(1-alpha)*x_start, alpha*x_end+(1-alpha)*x_start, 1)
        prophetPop = ea.Population(Encoding,Field,NIND,prophetChrom)
        """===========================调用算法模板进行种群进化========================"""
        # try:
        [BestIndi, population] = myAlgorithm.run(prophetPop)  # 执行算法模板，得到最优个体以及最后一代种群
        # except:
        #     print(myAlgorithm.population.CV.shape)
        """==================================输出结果=============================="""
        print('评价次数：%s' % myAlgorithm.evalsNum)
        print('时间已过 %s 秒' % myAlgorithm.passTime)
        if BestIndi.sizes != 0:
            print('最优的目标函数值为：%s' % BestIndi.ObjV[0][0])
            # if BestIndi.ObjV[0][0] > 10
            print('最优的控制变量值为：')
            p = BestIndi.Phen[0]
            it2fs = TrapezoidalIT2FS(
                p[0], p[1], p[2], p[3], p[4], p[4], p[5], p[6], p[7], p[8], p[9], p[9])
            print(it2fs)
            return it2fs
        else:
            print('没找到可行解。')
            return None



class SearchInRetainedRegion(ea.Problem):
    def __init__(self, retained_region, func, constraints, sense=1, NIND=50, MAXGEN=1000):
        m_l, m_u, coef, intercept, bias_l,bias_u = retained_region

        name = "Search in Retained Region"
        M = 1  # 目标维数
        maxormins = [sense]  # 目标最大最小（1为最小化，-1为最大化）
        if abs(m_u-m_l) > 1e-3 and abs(bias_u-bias_l) > 1e-3:
            lb = [m_l, bias_l]  # 决策变量下界
            ub = [m_u, bias_u]  # 决策变量上界
            self.vars = ['m','bias']
        elif abs(m_u-m_l) > 1e-3 and abs(bias_u-bias_l) <= 1e-3:
            lb = [m_l]
            ub = [m_u]
            self.vars = ['m']
        elif abs(m_u-m_l) <= 1e-3 and abs(bias_u-bias_l) > 1e-3:
            lb = [bias_l]
            ub = [bias_u]
            self.vars = ['bias']
        else:
            lb = []
            ub = []
            self.vars = []
        # print(lb,ub,self.vars)
        Dim = len(self.vars)

        lbin = [0] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [0] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        self.retained_region = retained_region
        self.NIND = NIND
        self.MAXGEN = MAXGEN
        self.func = func
        self.constraints = constraints

        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        m_l, m_u, coef, intercept, residuals_std, ks = self.retained_region
        Phen = pop.Phen
        
        if len(self.vars) == 2:
            m = Phen[:, [0]]
            bias = Phen[:, [1]]
            s = coef * m + intercept + bias
        elif self.vars[0] == "m":
            m = Phen[:, [0]]
            s = coef * m + intercept
        elif self.vars[0] == 'bias':
            m = ones_like(Phen)*(m_l+m_u)/2
            bias = Phen[:, [0]]
            s = coef * m + intercept + bias
        

        pop.ObjV = array([[self.func(m[i, 0], s[i, 0])]
                         for i in range(self.NIND)])

        pop.CV = hstack([
        ] + self.constraints(self.retained_region, m, s))

    def search(self):
        problem = self
        m_l, m_u, coef, intercept, residuals_std, ks = self.retained_region
        if len(self.vars) >= 1:
            
            Encoding = 'RI'
            NIND = self.NIND
            Field = ea.crtfld(Encoding, problem.varTypes,
                            problem.ranges, problem.borders)
            population = ea.Population(Encoding, Field, NIND)
            myAlgorithm = ea.soea_SEGA_templet(problem, population)
            myAlgorithm.MAXGEN = self.MAXGEN
            myAlgorithm.verbose = False
            myAlgorithm.trappedValue = 1e-10
            myAlgorithm.maxTrappedCount = 20
            myAlgorithm.drawing = 0
            [BestIndi, population] = myAlgorithm.run()
            if BestIndi.sizes != 0:
                # print("存在可行解！")
                best = BestIndi.ObjV[0][0]
                if len(self.vars) == 2:
                    # print("m,s均可变")
                    m, bias = BestIndi.Phen[0]
                    s = coef*m + intercept + bias
                    return best, [m, s]
                elif self.vars[0] == "m":
                    # print("仅m可变")
                    m = BestIndi.Phen[0][0]
                    bias = 0
                    s = coef*m + intercept + bias
                    return best, [m, s]
                elif self.vars[0] == "bias":
                    # print("仅s可变")
                    m = (m_l+m_u)/2
                    bias = BestIndi.Phen[0][0]
                    # print(BestIndi.Phen,BestIndi.CV)
                    s = coef * m + intercept + bias
                    return best, [m, s]
            else:
                # print("不存在可行解")
                m = (m_l+m_u)/2
                bias = 0
                s = coef*m+intercept + bias
                best = self.func(m,s)
                return best,(m,s)
        else:
            # print("m,s均不可变")
            m = (m_l+m_u)/2
            bias = 0
            s = coef*m+intercept + bias
            best = self.func(m,s)
            return best,(m,s)



class RetainedRegionApproach(Encoder):
    def __init__(self, l, r, M=10, penalty_factor=0.8, penalty_weights=[0.5, 0.5]):
        self.data = DataFrame(
            {"left": l, "right": r, "length": [i - j for i, j, in zip(r, l)]})
        self.M = M
        self.penalty_factor = penalty_factor
        self.penalty_weights = penalty_weights

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
        # self.tolerance_limit_processing()
        self.reasonable_interval_processing()

    # def fou_classify(self,models):
    #     self.fou_shape = fou_classify_hma(self.data,self.M)
    #     self.intra = models[self.fou_shape]
    #     return self.fou_shape

    
    def fou_classify(self,models,penalty_factor=1,penalty_weights=[0.5,0.5]):
        retained_region = self.retained_region_eval(penalty_factor,penalty_weights)
        self.intra = interior = models["INTERIOR"]
        al,_ = SearchInRetainedRegion(retained_region,interior.left_ms,unbounded_trap_constraints(self.intra), 1).search()
        bu,_ = SearchInRetainedRegion(retained_region,interior.right_ms,unbounded_trap_constraints(self.intra), -1).search()
        flag1 = al >= 0
        flag2 = bu <= self.M
        fou_shape = int(flag1) + 2* int(flag2)
        if fou_shape == 0:
            fou_shape = 3
        enum = ["NO FOU","RIGHT-SHOULDER","LEFT-SHOULDER","INTERIOR"]
        return enum[fou_shape]

    # def __set_intrapersonal_model(self, model):
    #     self.intra = model
    #     self.data['Embedded FS'] = [self.intra.interval_to_t1fs([row.left, row.right]) for i, row in
    #                                 self.data.iterrows()]

    # def intrapersonal_model_select(self,models):
    #     compatibilities = []
    #     for model in models:
    #         self.__set_intrapersonal_model(model)
    #         compatibilities.append(compatibility_riemann(self.data))
    #     best_model = models[argsort(compatibilities)[-1]]
    #     self.__set_intrapersonal_model(best_model)
    #     return {models[i]: compatibilities[i] for i in range(len(models))}


    @cached_property
    def relation_between_m_s(self):
        left = self.data.left.to_numpy().reshape((-1, 1))
        right = self.data.right.to_numpy().reshape((-1, 1))
        m = (left+right)/2
        s = (right-left)/sqrt(12)
        reg = LinearRegression(fit_intercept=True).fit(m, s)
        coef, intercept = reg.coef_[0][0], reg.intercept_[0]
        def f_s(x): return coef*x + intercept
        s_fitted = f_s(left)
        residuals = s - s_fitted
        residuals_std = std(residuals)
        return coef, intercept, residuals

    # @cached_property
    def retained_region_eval(self,penalty_factor,penalty_weights):
        if not penalty_factor:
            penalty_factor = self.penalty_factor
        if not penalty_weights:
            penalty_weights = self.penalty_weights
        penalty_factor_m = 1 - (1-penalty_factor) ** (penalty_weights[0])
        penalty_factor_s = 1 - (1-penalty_factor) ** (penalty_weights[1])
        # print(penalty_factor,penalty_factor_m,penalty_factor_s)

        left = self.data.left.to_numpy().reshape((-1, 1))
        right = self.data.right.to_numpy().reshape((-1, 1))
        m = (left+right)/2
        s = (right-left)/sqrt(12)

        coef, intercept, residuals = self.relation_between_m_s

        m_u = quantile(m, 1 - (penalty_factor_m)/2)
        m_l = quantile(m, (penalty_factor_m)/2)
        # bias_u = quantile(residuals, 1 - (penalty_factor_s)/2)
        # bias_l = quantile(residuals, (penalty_factor_s)/2)
        ks = norm.ppf(1-penalty_factor_s/2)
        bias_u =  ks * std(residuals)
        bias_l =  -ks * std(residuals)
        # km = norm.ppf(1-(penalty_factor_m)/2)
        # m_u = mean(m) + km * std(m)
        # m_l = mean(m) - km * std(m)

        return m_l, m_u, coef, intercept, bias_l, bias_u

    @cached_property
    def retained_region(self):
        return self.retained_region_eval(self.penalty_factor,self.penalty_weights)

    def set_intrapersonal_model(self, models):
        # self.fou_shape = self.fou_classify(models,self.penalty_factor,self.penalty_weights)
        self.fou_shape = self.fou_classify(models)
        # print(self.fou_shape)
        self.intra = models[self.fou_shape]
        self.data['Embedded FS'] = [self.intra.interval_to_t1fs([row.left, row.right]) for i, row in
                                    self.data.iterrows()]
    
    @cached_property
    def retained_region_FOU(self,reduction=True):
        l_l,_ = SearchInRetainedRegion(
            self.retained_region, self.intra.left_ms, trap_constraints(self.intra), 1).search()
        r_u,_ = SearchInRetainedRegion(
            self.retained_region, self.intra.right_ms, trap_constraints(self.intra), -1).search()
        x = linspace(l_l, r_u, 100)
        lmf = []
        umf = []
        for xi in x:
            mu_xi = mu_trap_x(xi, self.intra)
            lmf.append(SearchInRetainedRegion(self.retained_region,
                       mu_xi, trap_constraints(self.intra), 1).search()[0])
            umf.append(SearchInRetainedRegion(self.retained_region,
                       mu_xi, trap_constraints(self.intra), -1).search()[0])
        x, lmf, umf = x, array(lmf), array(umf)
        if reduction:
            N = 100
            index = umf > 0
            x_start = x[index].min()
            x_end = x[index].max()
            L = x_end - x_start
            x_start -= 0.1*L
            x_end += 0.1*L
            x_start = max([x_start, 0])
            x_end = min([x_end, self.M])
            x = linspace(x_start, x_end, N)
            for i,xi in enumerate(x):
                mu_xi = mu_trap_x(xi, self.intra)
                lmf[i] = SearchInRetainedRegion(self.retained_region,
                        mu_xi, trap_constraints(self.intra), 1).search()[0]
                umf[i] = SearchInRetainedRegion(self.retained_region,
                        mu_xi, trap_constraints(self.intra), -1).search()[0]
            return x, array(lmf), array(umf)
        else:
            return x, array(lmf), array(umf)
    
    @cached_property
    def IT2FS_ga(self):
        x,lmf,umf = self.retained_region_FOU
        problem = TrapezoidalIT2FS_constructor(x, lmf, umf)
        return problem.solve()
    
    @cached_property
    def IT2FS_eia(self):
        am,_ = SearchInRetainedRegion(
            self.retained_region, self.intra.left_ms, trap_constraints(self.intra), 1).search()
        aM,_ = SearchInRetainedRegion(
            self.retained_region, self.intra.left_ms, trap_constraints(self.intra), -1).search()
        am = am if am > 0 else 0
        aM = aM if aM > 0 else 0
        if am > aM:
            am,aM = aM,am
        bm,_ = SearchInRetainedRegion(
            self.retained_region, self.intra.mid_left_ms, trap_constraints(self.intra), 1).search()
        bM,_ = SearchInRetainedRegion(
            self.retained_region, self.intra.mid_left_ms, trap_constraints(self.intra), -1).search()
        if bm > bM:
            bm,bM = bM,bm
        cm,_ = SearchInRetainedRegion(
            self.retained_region, self.intra.mid_right_ms, trap_constraints(self.intra), 1).search()
        cM,_ = SearchInRetainedRegion(
            self.retained_region, self.intra.mid_right_ms, trap_constraints(self.intra), -1).search()
        if cm > cM:
            cm,cM = cM,cm
        dm,_ = SearchInRetainedRegion(
            self.retained_region, self.intra.right_ms, trap_constraints(self.intra), 1).search()
        dM,_ = SearchInRetainedRegion(
            self.retained_region, self.intra.right_ms, trap_constraints(self.intra), -1).search()
        dm = dm if dm < self.M else self.M
        dM = dM if dM < self.M else self.M
        if dm > dM:
            dm,dM = dM,dm
        if aM >= dm:
            return TrapezoidalIT2FS(am, bm, cM, dM, 1, 1, (aM+dm)/2,(aM+dm)/2,(aM+dm)/2,(aM+dm)/2,0,0)
        elif bM > cm:
            p = (dm*(bM-aM)+aM*(dm-cm))/((bM-aM)+(dm-cm))
            mu_p = (dm-p)/(dm-cm)
            if not isnan(mu_p):
                return TrapezoidalIT2FS(am, bm, cM, dM, 1, 1, aM, p,p, dm, mu_p, mu_p)
            else:
                return TrapezoidalIT2FS(am, bm, cM, dM, 1, 1, aM, bM, cm, dm, 1, 1)
        else:
            return TrapezoidalIT2FS(am, bm, cM, dM, 1, 1, aM, bM, cm, dm, 1, 1)
    
    @cached_property
    def IT2FS(self):
        return self.IT2FS_eia
