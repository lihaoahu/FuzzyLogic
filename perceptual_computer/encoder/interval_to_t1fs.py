from abc import abstractmethod

from numba import jit,prange
from numpy import array, float64, hstack, repeat, zeros,abs,zeros,float64,sqrt,zeros_like,ones_like
from numpy.random import random,normal

from fuzzy_logic.type_1.fuzzy_sets import Type1FuzzySet,TriangularFuzzyNumber,TrapezoidalFuzzySet,TrapezoidalFuzzyNumber

SQRT_6 = sqrt(6)
SQRT_3 = sqrt(3)
SQRT_2 = sqrt(2)
SQRT_15 = sqrt(15)

class IntervalToType1FuzzySet:
    name = 'Mapping from interval to fuzzy set'

    @abstractmethod
    def interval_to_t1fs(self,interval):
        return Type1FuzzySet()

    def __call__(self,interval):
        return self.interval_to_t1fs(interval)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class SymmetricTriangularModel(IntervalToType1FuzzySet):
    name = 'Symmetrical triangular model'

    @staticmethod
    def interval_to_t1fs(interval):
        l,r = interval
        a = (l+r)/2 - (r-l)/sqrt(2)
        b = (l+r)/2 + (r-l)/sqrt(2)
        return TriangularFuzzyNumber(a,(a+b)/2,b)
    
    @staticmethod
    @jit(nopython=True)
    def left_ms(m,s):
        return m-SQRT_6*s
    
    @staticmethod
    @jit(nopython=True)
    def right_ms(m,s):
        return m+SQRT_6*s
    
    @staticmethod
    @jit(nopython=True)
    def mid_left_ms(m,s):
        return m
    
    @staticmethod
    @jit(nopython=True)
    def mid_right_ms(m,s):
        return m


class SymmetricTrapezoidalModel(IntervalToType1FuzzySet):
    name = 'Symmetrical trapezoidal model'

    @staticmethod
    def interval_to_t1fs(interval):
        l,r = interval
        a = (l+r)/2 - 3*sqrt(5)*(r-l)/10
        b = (l+r)/2 + 3*sqrt(5)*(r-l)/10
        return TrapezoidalFuzzyNumber(a,(2*a+b)/3,(a+2*b)/3,b)

    @staticmethod
    @jit(nopython=True)
    def left_ms(m,s):
        return m-3*SQRT_15*s/5
    
    @staticmethod
    @jit(nopython=True)
    def right_ms(m,s):
        return m+3*SQRT_15*s/5
    
    @staticmethod
    @jit(nopython=True)
    def mid_left_ms(m,s):
        return m - SQRT_15*s/5
    
    @staticmethod
    @jit(nopython=True)
    def mid_right_ms(m,s):
        return m + SQRT_15*s/5


class SymmetricRectangularModel(IntervalToType1FuzzySet):
    name = 'Symmetrical rectangular model'

    @staticmethod
    def interval_to_t1fs(interval):
        l,r = interval
        a = l
        b = r
        return TrapezoidalFuzzyNumber(a,a,b,b)

    @staticmethod
    @jit(nopython=True)
    def left_ms(m,s):
        return m - SQRT_3*s
    
    @staticmethod
    @jit(nopython=True)
    def right_ms(m,s):
        return m + SQRT_3*s
    
    @staticmethod
    @jit(nopython=True)
    def mid_left_ms(m,s):
        return m - SQRT_3*s
    
    @staticmethod
    @jit(nopython=True)
    def mid_right_ms(m,s):
        return m + SQRT_3*s
    



class LeftShoulderTrapezoidalModel(IntervalToType1FuzzySet):
    name = 'Left-shoulder trapezoidal model'

    @staticmethod
    def interval_to_t1fs(interval):
        l,r = interval
        a = (l+r)/2 - (r-l)/sqrt(6)
        b = (l+r)/2 + sqrt(6/9)*(r-l)
        return TrapezoidalFuzzySet(0,0,a,b,1,1)
    
    @staticmethod
    @jit(nopython=True)
    def left_ms(m,s):
        return zeros_like(m)
    
    @staticmethod
    @jit(nopython=True)
    def right_ms(m,s):
        return m+2*SQRT_2*s
    
    @staticmethod
    @jit(nopython=True)
    def mid_left_ms(m,s):
        return zeros_like(m)
    
    @staticmethod
    @jit(nopython=True)
    def mid_right_ms(m,s):
        return m-SQRT_2*s


class RightShoulderTrapezoidalModel(IntervalToType1FuzzySet):
    name = 'Right-shoulder trapezoidal model'

    def __init__(self, M):
        self.M = M
    
    def interval_to_t1fs(self, interval):
        l, r = interval
        a = (l + r) / 2 - sqrt(6 / 9) * (r - l)
        b = (l + r) / 2 + (r - l) / sqrt(6)
        return TrapezoidalFuzzySet(a, b, self.M, self.M,1,1)
    
    @staticmethod
    @jit(nopython=True)
    def left_ms(m,s):
        return m-2*SQRT_2*s
    
    # @jit(nopython=True)
    def right_ms(self,m,s):
        return self.M*ones_like(m)
    
    @staticmethod
    @jit(nopython=True)
    def mid_left_ms(m,s):
        return m+SQRT_2*s
    
    # @jit(nopython=True)
    def mid_right_ms(self,m,s):
        return self.M*ones_like(m)


class LeftTruncatedTriangularModel(IntervalToType1FuzzySet):
    name = 'Left-truncated triangular model'

    def __init__(self,apex):
        self.apex= apex

    def interval_to_t1fs(self, interval):
        l,r = interval
        a = self.apex
        b = self.apex + sqrt(3/2)*(r-self.apex)
        left_height = max([1 - a/(b-a),0])
        left = left_apex = max([2*a - b,0])
        return TrapezoidalFuzzySet(left,left_apex,a,b,left_height,1)


class RightTruncatedTriangularModel(IntervalToType1FuzzySet):
    name = 'Right-shoulder trapezoidal model'

    def __init__(self, M, apex):
        self.M = M
        self.apex = apex

    def interval_to_t1fs(self, interval):
        l,r = interval
        a = self.apex - sqrt(3 / 2) * (self.apex - l)
        b = self.apex
        right_height = max([(2*b-a-self.M)/(b-a),0])
        right = right_apex = min([2*b - a,self.M])
        return TrapezoidalFuzzySet(a,b,right_apex,right,1,right_height)


class ApexFixedTrapezoidalModel(IntervalToType1FuzzySet):
    name = 'Apex fixed trapezoidal model'

    def __init__(self,apexes):
        self.left_apex,self.right_apex = apexes

    def interval_to_t1fs(self, interval):
        l, r = interval
        if l <= self.left_apex and r >= self.right_apex:
            a = self.left_apex - sqrt(3/2)*(self.left_apex-l)
            b = self.right_apex + sqrt(3/2)*(r-self.right_apex)
            return TrapezoidalFuzzySet(a, self.left_apex, self.right_apex, b,1,1)
        elif l <= self.left_apex:
            a = self.left_apex - sqrt(3 / 2) * (self.left_apex - l)
            return TrapezoidalFuzzySet(a, self.left_apex, self.right_apex, self.right_apex, 1,1)
        elif r >= self.right_apex:
            b = self.right_apex + sqrt(3/2)*(r-self.right_apex)
            return TrapezoidalFuzzySet(self.left_apex, self.left_apex, self.right_apex, b, 1,1)
        else:
            return TrapezoidalFuzzySet(self.left_apex, self.left_apex, self.right_apex, self.right_apex, 1,1)

    def __str__(self):
        return f"Apexes-fixed ({self.left_apex:.2f},{self.right_apex:.2f}) trapezoidal model"