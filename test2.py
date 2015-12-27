import sys
import numpy as np
from scipy.stats import describe
#from scipy.stats.mstats import zscore
from scipy.stats import norm as normal
from sklearn import datasets
from sklearn.metrics import roc_auc_score
from ComputeBart import ComputeBart

digits = datasets.load_digits()

sel_7_vs_9 = (digits.target==7) | (digits.target==9)

x = digits.data[sel_7_vs_9,:]
y = digits.target[sel_7_vs_9]

print '7 vs 9 dataset dims:', x.shape

target = np.array([3.0 if v==9 else -3.0 for v in y])

bart = ComputeBart(regression=False)
result = bart.fit_and_predict(x, target, x)

standard_normal = normal()

probs = np.vectorize(standard_normal.cdf)(result)
target = np.array([True if v==9 else False for v in y], dtype=np.bool)

print 'Targets:'
print target

print 'Probs:'
print probs

print 'AUC =', roc_auc_score(target, probs)
