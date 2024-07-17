from functools import cached_property

from numba import jit
from numpy import array, exp
from numpy.lib.function_base import iterable

@jit(nopython=True)
def triangular_mf(x,l, m, u):
    if l < x <= m:
        return 1-(m - x)/(m - l)
    elif m < x < u:
        return (u - x)/(u-m)
    else:
        return 0

@jit(nopython=True)
def trapezoidal_mf(x, left, mid_left, mid_right, right, height_left, height_right):
    if left <= x < mid_left:
        return height_left*(x-left)/(mid_left-left)
    elif mid_left <= x <= mid_right:
        if mid_left == mid_right:
            return (height_left + height_right)/2
        return height_left + (height_right-height_left) * (x-mid_left)/(mid_right-mid_left)
    elif mid_right < x <= right:
        return height_right*(right-x)/(right-mid_right)
    else:
        return 0


class Type1FuzzySet:
    def __call__(self, x):
        return self.membership(x)
    


class TriangularFuzzyNumber(Type1FuzzySet):
    def __init__(self, l, m, u):
        assert self.verify(l,m,u)
        self.left = l
        self.right = u
        if m!=None:
            self.middle = m
        else:
            self.middle = (l + u) / 2
    
    @classmethod
    def verify(cls,l,m,u):
        return l<=m<=u

    @cached_property
    def membership(self):
        
        def fun(x):
            if iterable(x):
                return array([triangular_mf(i, self.left, self.middle, self.right) for i in x], dtype=float)
            else:
                return triangular_mf(x, self.left, self.middle, self.right)
        return fun
    
    def alpha_cut(self,alpha):
        return [self.left + alpha * (self.middle - self.left), self.right - alpha * (self.right - self.middle)]
    
    @cached_property
    def _S_Y(self):
        if self.left >= 0:
            return (self.left + 2 * self.middle + self.right)/2
        elif self.middle >= 0:
            return (self.right + self.middle)/2 + self.middle**2/2/(self.middle - self.left)
        elif self.right >= 0:
            return self.right**2/2/(self.right - self.middle)
        else:
            return 0
    
    def __add__(self, other):
        return TriangularFuzzyNumber(self.left + other.left, self.middle + other.middle, self.right + other.right)

    def __sub__(self, other):
        return TriangularFuzzyNumber(self.left - other.right, self.middle - other.middle, self.right - other.left)

    def __mul__(self, other):
        if isinstance(other,TriangularFuzzyNumber):
            return TriangularFuzzyNumber(self.left * other.left, self.middle * other.middle, self.right * other.right)
        elif other >= 0:
            return TriangularFuzzyNumber(self.left * other, self.middle * other, self.right * other)
        else:
            raise TypeError(f"未定义的乘法：\n{self} x {other}")

    def __truediv__(self, other):
        return TriangularFuzzyNumber(self.left / other.right, self.middle / other.middle, self.right / other.left)

    @cached_property
    def center_of_area(self):
        return (self.right ** 2 - self.left ** 2 + self.middle * (self.right - self.left))/(3 * (self.right - self.left))

    def __str__(self):
        return f"Triangular<{self.left:.2f},{self.middle:.2f},{self.right:.2f}>"

    def __repr__(self):
        return self.__str__()


class TrapezoidalFuzzyNumber(Type1FuzzySet):
    def __init__(self, left, mid_left, mid_right, right):
        self.left = left
        self.right = right
        self.mid_left, self.mid_right = mid_left, mid_right

    @cached_property
    def membership(self):

        def fun(x):
            if iterable(x):
                return array([trapezoidal_mf(i, self.left, self.mid_left, self.mid_right, self.right,1,1) for i in x], dtype=float)
            else:
                return trapezoidal_mf(x, self.left, self.mid_left, self.mid_right, self.right,1,1)

        return fun

    def __str__(self):
        return f"Trapezoidal<{self.left:.2f},{self.mid_left:.2f},{self.mid_right:.2f},{self.right:.2f}>"

    def __repr__(self):
        return self.__str__()


class TrapezoidalFuzzySet(Type1FuzzySet):
    def __init__(self, left, mid_left, mid_right, right, height_left, height_right):
        self.left = left
        self.right = right
        self.mid_left, self.mid_right = mid_left, mid_right
        self.height_left, self.height_right = height_left, height_right

    @cached_property
    def membership(self):
        def fun(x):
            if iterable(x):
                return array([trapezoidal_mf(i, self.left, self.mid_left, self.mid_right, self.right, self.height_left, self.height_right) for i in x], dtype=float)
            else:
                return trapezoidal_mf(x, self.left, self.mid_left, self.mid_right, self.right, self.height_left, self.height_right)
        return fun

    def __str__(self):
        return f"Trapezoidal<{self.left:.2f},{self.mid_left:.2f}({self.height_left:.2f}),{self.mid_right:.2f}({self.height_right:.2f}),{self.right:.2f}>"

    def __repr__(self):
        return self.__str__()
