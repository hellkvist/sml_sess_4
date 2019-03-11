import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb

x = np.arange(1,41)[:,np.newaxis]
y = np.zeros(shape=(40,1))
y[np.array([34, 38, 39, 40])-1] = 1
reg = skl_lm.LinearRegression().fit(x, y)
plt.plot(x, reg.predict(x), label='lin reg')

logreg = skl_lm.LogisticRegression(solver='lbfgs', C=1000).fit(x, y.flatten())
proba = logreg.predict_proba(x)
plt.plot(x, logreg.predict(x), label='log reg')
plt.plot(x, logreg.predict_proba(x), label='log reg probas')
plt.plot(x, y, '*', label='true y:s')
plt.show()