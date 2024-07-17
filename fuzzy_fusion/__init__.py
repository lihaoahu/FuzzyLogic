from fuzzy_logic.type_1.fuzzy_sets import TriangularFuzzyNumber
from numpy import reshape

def TriFN_WA(TriFNs,weights):
    TriFNs = reshape(TriFNs,(-1,))
    weights = reshape(weights,(-1,))
    N = len(TriFNs)
    assert len(weights) == N
    return (TriFNs*weights).sum()/weights.sum()
