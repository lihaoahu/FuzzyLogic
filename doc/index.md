# FuzzyLogic

## Introduction
This repository implements servral computing with words encoders for encoding words into interval type-2 fuzzy sets.
- Interval approach
- Enhanced interval approach
- Hao-Mendel approach
- Retained region approach
- Least-squares framework

## Usages
```python
import numpy as np

from perceptual_computer.data import intervals_175subjects as data
from fuzzy_logic.interval_type_2.type_reduce import KarnikMendel

word_list = ["Teeny-weeny", "Tiny", "None to very little",
             "A smidgen", "Very small", "Very little",
             "A bit", "Little", "Low amount", "Small",
             "Somewhat small", "Some", "Quite a bit",
             "Modest amount", "Some to moderate",
             "Medium", "Moderate amount", "Fair amount",
             "Good amount", "Considerable amount",
             "Sizeable", "Substantial amount", "Large",
             "Very sizeable", "A lot", "High amount",
             "Very large", "Very high amount", "Huge amount",
             "Humongous amount", "Extreme amount", "Maximum amount"]

from perceptual_computer.encoder.least_squares_framework import LeastSquaresFramework,compatibility_riemann,fuzzy_statistic,interrelated_degree
from perceptual_computer.encoder.interval_to_t1fs import (SymmetricTriangularModel, LeftShoulderTrapezoidalModel, 
        RightShoulderTrapezoidalModel, SymmetricTrapezoidalModel, SymmetricRectangularModel,ApexFixedTrapezoidalModel)

lsf_dict = dict()
for word in data.keys():
    endpoints= data[word]
    lsf = LeastSquaresFramework(
        l=endpoints.left, r=endpoints.right,M=1
    )
    models = [
        ApexFixedTrapezoidalModel(apexes=[0,min(lsf.data.right)]),
        ApexFixedTrapezoidalModel(apexes=[max(lsf.data.left),min(lsf.data.right)]),
        ApexFixedTrapezoidalModel(apexes=[max(lsf.data.left),1]),
        LeftShoulderTrapezoidalModel(),
        SymmetricTriangularModel(),
        RightShoulderTrapezoidalModel(M=10),
    ]
    lsf.data_part()
    lsf.intrapersonal_model_select(models)
    lsf_dict[word] = lsf

it2fs_dict = dict()
for word in data.keys():
    it2fs_dict[word] = lsf_dict[word].IT2FS(alpha=0.2)
```