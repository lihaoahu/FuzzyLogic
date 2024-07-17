from abc import ABCMeta, abstractmethod, abstractproperty
from functools import cached_property
from numpy import sqrt
from fuzzy_logic.type_1.fuzzy_sets import TrapezoidalFuzzySet


class IntervalType2FuzzySet(metaclass=ABCMeta):

    @abstractproperty
    def lower_mf(self):
        pass

    @abstractproperty
    def upper_mf(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()


class TrapezoidalIT2FS(IntervalType2FuzzySet):
    def __init__(self, umf_left, umf_left_apex, umf_right_apex, umf_right, umf_left_height, umf_right_height, lmf_left, lmf_left_apex, lmf_right_apex, lmf_right, lmf_left_height, lmf_right_height):
        self.umf_left, self.umf_left_apex, self.umf_right_apex, self.umf_right, self.umf_left_height, self.umf_right_height = umf_left, umf_left_apex, umf_right_apex, umf_right, umf_left_height, umf_right_height
        self.lmf_left, self.lmf_left_apex, self.lmf_right_apex, self.lmf_right, self.lmf_left_height, self.lmf_right_height = lmf_left, lmf_left_apex, lmf_right_apex, lmf_right, lmf_left_height, lmf_right_height

    @cached_property
    def lower_mf(self):
        return TrapezoidalFuzzySet(self.lmf_left, self.lmf_left_apex, self.lmf_right_apex, self.lmf_right, self.lmf_left_height, self.lmf_right_height)

    @cached_property
    def upper_mf(self):
        return TrapezoidalFuzzySet(self.umf_left, self.umf_left_apex, self.umf_right_apex, self.umf_right, self.umf_left_height, self.umf_right_height)

    @cached_property
    def area(self):
        upper = (self.umf_right - self.umf_left + self.umf_right_apex - self.umf_left_apex)*self.umf_right_height / \
            2 + (self.umf_right_apex-self.umf_left) * \
            (self.umf_left_height-self.umf_right_height)/2
        lower = (self.lmf_right - self.lmf_left + self.lmf_right_apex - self.lmf_left_apex)*self.lmf_right_height / \
            2 + (self.lmf_right_apex-self.lmf_left) * \
            (self.lmf_left_height-self.lmf_right_height)/2
        return abs(upper - lower)

    def __str__(self):
        return f"<{self.umf_left:.2f},{self.umf_left_apex:.2f},{self.umf_right_apex:.2f},{self.umf_right:.2f},{self.umf_left_height:.2f},{self.umf_right_height:.2f}>" + f"<{self.lmf_left:.2f},{self.lmf_left_apex:.2f},{self.lmf_right_apex:.2f},{self.lmf_right:.2f},{self.lmf_left_height:.2f},{self.lmf_right_height:.2f}>"

    def __repr__(self):
        return self.__str__()


class Chen_TrapezoidalIT2FS(TrapezoidalIT2FS):

    def __init__(self, umf_left, umf_left_apex, umf_right_apex, umf_right, umf_height, lmf_left, lmf_left_apex, lmf_right_apex, lmf_right, lmf_height):
        self.umf_left, self.umf_left_apex, self.umf_right_apex, self.umf_right, self.umf_left_height, self.umf_right_height = umf_left, umf_left_apex, umf_right_apex, umf_right, umf_height, umf_height
        self.lmf_left, self.lmf_left_apex, self.lmf_right_apex, self.lmf_right, self.lmf_left_height, self.lmf_right_height = lmf_left, lmf_left_apex, lmf_right_apex, lmf_right, lmf_height, lmf_height

    def rank_value(self):
        r1 = (
                (self.umf_left+self.umf_right)/2 + 
                (self.umf_left_height+self.lmf_left_height)/4
            )*(
                (self.lmf_left+self.umf_left) +
                (self.lmf_left_apex+self.umf_left_apex)+
                (self.lmf_right_apex+self.umf_right_apex)+
                (self.lmf_right+self.umf_right)
            )/8
        # r2 = (
        #         (self.umf_left+self.umf_right)**(1/2)+
        #         (self.lmf_left_height*self.umf_left_height)**(1/4)
        #     )*(
        #         (self.lmf_left*self.umf_left)* 
        #         (self.lmf_left_apex*self.umf_left_apex)*
        #         (self.lmf_right_apex*self.umf_right_apex)*
        #         (self.lmf_right*self.umf_right)
        #     )**(1/8)
        # r3 = 16*(
        #         self.umf_left*self.umf_right/(self.umf_left+self.umf_right)+
        #         self.lmf_left_height*self.umf_left_height/(self.lmf_left_height+self.umf_left_height)
        #     )/(
        #         (1/self.lmf_left+1/self.umf_left)+
        #         (1/self.lmf_left_apex+1/self.umf_left_apex) +
        #         (1/self.lmf_right_apex+1/self.umf_right_apex)+
        #         (1/self.lmf_right+1/self.umf_right)
        #     )
        # alpha = 
        return r1

    def distance(self,other,lamb=0.5):
        def R(A):
            return 1 - A.lmf_right - \
            lamb*(A.lmf_left - A.umf_left + A.umf_right - A.lmf_right) - \
            1/(2*A.lmf_left_height*A.umf_left_height)*(
                A.umf_left_height*(lamb*(A.lmf_left_apex-A.lmf_left-A.umf_left_apex+A.umf_left)) -
                (A.lmf_right-A.lmf_right_apex-A.lmf_left_apex+A.lmf_left)-
                A.lmf_left_height*(A.umf_right-A.umf_right_apex-A.lmf_right+A.lmf_right_apex)
            )
        r1 = R(self)
        r2 = R(other)
        return abs(r1-r2)

    def __add__(self, other):
        return Chen_TrapezoidalIT2FS(
            self.umf_left + other.umf_left,
            self.umf_left_apex + other.umf_left_apex,
            self.umf_right_apex + other.umf_right_apex,
            self.umf_right + other.umf_right,
            min(self.umf_left_height, other.umf_left_height),
            min(self.umf_right_height, other.umf_right_height),
            self.lmf_left + other.lmf_left,
            self.lmf_left_apex + other.lmf_left_apex,
            self.lmf_right_apex + other.lmf_right_apex,
            self.lmf_right + other.lmf_right,
            min(self.lmf_left_height, other.lmf_left_height),
            min(self.lmf_right_height, other.lmf_right_height)
        )

    def __sub__(self, other):
        return Chen_TrapezoidalIT2FS(
            self.umf_left - other.umf_right,
            self.umf_left_apex - other.umf_right_apex,
            self.umf_right_apex - other.umf_left_apex,
            self.umf_right - other.umf_left,
            min(self.umf_left_height, other.umf_left_height),
            min(self.umf_right_height, other.umf_right_height),
            self.lmf_left - other.lmf_right,
            self.lmf_left_apex - other.lmf_right_apex,
            self.lmf_right_apex - other.lmf_left_apex,
            self.lmf_right - other.lmf_left,
            min(self.lmf_left_height, other.lmf_left_height),
            min(self.lmf_right_height, other.lmf_right_height)
        )

    def __mul__(self, other):
        if isinstance(other, Chen_TrapezoidalIT2FS):
            pass
        else:
            other = Chen_TrapezoidalIT2FS(
                other,
                other,
                other,
                other,
                1, 1,
                other,
                other,
                other,
                other,
                1, 1
            )
        return Chen_TrapezoidalIT2FS(
            self.umf_left * other.umf_left,
            self.umf_left_apex * other.umf_left_apex,
            self.umf_right_apex * other.umf_right_apex,
            self.umf_right * other.umf_right,
            min(self.umf_left_height, other.umf_left_height),
            min(self.umf_right_height, other.umf_right_height),
            self.lmf_left * other.lmf_left,
            self.lmf_left_apex * other.lmf_left_apex,
            self.lmf_right_apex * other.lmf_right_apex,
            self.lmf_right * other.lmf_right,
            min(self.lmf_left_height, other.lmf_left_height),
            min(self.lmf_right_height, other.lmf_right_height)
        )
