import numpy as np
import bart


x = np.genfromtxt('x1.txt', delimiter=' ')
y = np.genfromtxt('y1.txt', delimiter=' ')

y1 = np.empty_like(y)

bart_classifier = bart.compute_bart()

bart_classifier.set_insample_matrix(x)
bart_classifier.set_insample_target(y)

bart_classifier.set_outsample_matrix(x)
bart_classifier.set_outsample_target(y1)

bart_classifier.fit()

print y1
