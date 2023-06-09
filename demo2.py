#

import numpy as np
import math

n = 3000 #
s = 150 # split to exterior validation set

m = 20

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

opt_grid = {"batch_size" : (1000,),
            "learning_rate": (1e-5, 1e-1),
            "num_epochs" : (100,150)}
m = modelList_torch_tsRegressor[0]
m2 = modelList_torch_tsRegressor[1]
m["opt"] = opt_grid
m2["opt"] = opt_grid


modelList=modelList + [m,m2]


from tuneSurvey.ts_torchLists import *

tscv = TimeSeriesSplit(n_splits=3)


import numpy as np
from tuneSurvey.tsVectorize import*

import os
#os.mkdir("vec_search")
#os.mkdir("tsNN_search")
vsearch_modelList([m],X,21,tscv,device)
#torch.cuda.empty_cache()
vsearch_modelList([m2],X,21,tscv,device)
##
##
# for sk grid
#real	241m17.713s
#user	1839m33.983s
#sys	2m39.234s


