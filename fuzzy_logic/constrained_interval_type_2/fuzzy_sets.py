from abc import ABCMeta, abstractmethod, abstractproperty
from functools import cached_property
from random import uniform
from numpy import vectorize,iterable

from fuzzy_logic.type_1.fuzzy_sets import TrapezoidalFuzzySet,Type1FuzzySet,TriangularFuzzyNumber


class ConstrainedIntervalType2FuzzySet(metaclass=ABCMeta):
    
    def __init__(self,C0,*C):
        assert issubclass(C0,Type1FuzzySet)
        self.C0 = C0
        
        self.n = len(C)
        for i in range(self.n):
            assert isinstance(C[i],list) and len(C[i]) == 2
        self.C = C
    
    def singly_sample(self):
        try:
            c = [uniform(self.C[i][0],self.C[i][1]) for i in range(self.n)]
            efs = self.C0(*c)
            return efs
        except:
            return self.singly_sample()

    def sample(self,N):
        return [self.singly_sample() for _ in range(N)]


class TriangularConstrainedIntervalType2FuzzySet(ConstrainedIntervalType2FuzzySet):
    def __init__(self,*C):
        super().__init__(TriangularFuzzyNumber,*C)
        self.left_min,self.left_max = C[0]
        self.mid_min,self.mid_max = C[1]
        self.right_min,self.right_max = C[2]
    
    @cached_property
    def upper_mf(self):
        def fun(x):
            if iterable(x):
                return [fun(xi) for xi in x]
            else:
                if x >= self.left_min and x <= self.mid_min:
                    return (x-self.left_min)/(self.mid_min-self.left_min)
                elif x > self.mid_min and x < self.mid_max:
                    return 1
                elif x >= self.mid_max and x <= self.right_max:
                    return (self.right_max - x)/(self.right_max - self.mid_max)
                else:
                    return 0
        return fun

    @cached_property
    def lower_mf(self):
        def fun(x):
            if iterable(x):
                return [fun(xi) for xi in x]
            else:
                if x < self.left_max or x > self.right_min:
                    return 0
                else:
                    v1 = (x-self.left_max)/(self.mid_max-self.left_max)
                    v2 = (self.right_min-x)/(self.right_min-self.mid_min)
                    if v1 >= v2:
                        return v2
                    else:
                        return v1
        return fun

    def central_efn(self):
        return TriangularFuzzyNumber(
            (self.left_max+self.left_min)/2,
            (self.mid_max+self.mid_min)/2,
            (self.right_max+self.right_min)/2
        )
    