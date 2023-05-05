#

import numpy as np
import math

n = 20000 #
s = 150 # split to exterior validation set

m = 10

X = []
for i in range(n):
    
    X.append([math.sin(i) + math.cos(j) for j in range(m)])

X = np.array(X)
from tuneSurvey.skLists import modelList_sklearn_regressor_lite

modelList = modelList_sklearn_regressor_lite[2:-2]

from tuneSurvey.boostingLists import *
from tuneSurvey.ts_torchLists import *
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )



modelList=modelList + [boostingr_grid[1]]


modelList=modelList + [modelList_torch_tsRegressor[0]]


from tuneSurvey.ts_torchLists import *

tscv = TimeSeriesSplit(n_splits=3)


import numpy as np
from tuneSurvey.tsVectorize import*

import os
os.mkdir("vec_search")
os.mkdir("tsNN_search")
vsearch_modelList([modelList_torch_tsRegressor[0]],X,14,tscv,device)

