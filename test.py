import numpy as np
import bart
from ComputeBart import ComputeBart


x = np.genfromtxt('x1.txt', delimiter=' ')
y = np.genfromtxt('y1.txt', delimiter=' ')

y1 = np.empty_like(y)

# using bart.compute_bart directly
bart_classifier = bart.compute_bart()

bart_classifier.set_insample_matrix(x)
bart_classifier.set_insample_target(y)

bart_classifier.set_outsample_matrix(x)
bart_classifier.set_outsample_target(y1)

bart_classifier.fit()

print y1

#using ComputeBart wrapper
bart = ComputeBart()
y1 = bart.fit_and_predict(x,y,x)

print y1
