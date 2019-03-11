import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb

np.random.seed(1)
iris = pd.read_csv('../Data/iris.csv')

cols_in = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
cols_out = ['Species']
N = len(iris)
n_train = round(N/2)
n_test = N - n_train

idx_bool = iris.index.isin(np.random.choice(N, size=(n_train,), replace=False))
train = iris.iloc[idx_bool]
test = iris.iloc[~idx_bool]

LDA = skl_da.LinearDiscriminantAnalysis().fit(train[cols_in], train[cols_out].values.flatten())
QDA = skl_da.QuadraticDiscriminantAnalysis().fit(train[cols_in], train[cols_out].values.flatten())
kNN = skl_nb.KNeighborsClassifier(n_neighbors=3).fit(train[cols_in], train[cols_out].values.flatten())

LDA_hat = LDA.predict(test[cols_in])
QDA_hat = QDA.predict(test[cols_in])
kNN_hat = kNN.predict(test[cols_in])

print(pd.crosstab(test[cols_out].values.flatten(), LDA_hat, rownames=['True values'], colnames=['LDA']), '\n')
print(pd.crosstab(test[cols_out].values.flatten(), QDA_hat, rownames=['True values'], colnames=['QDA']), '\n')
print(pd.crosstab(test[cols_out].values.flatten(), kNN_hat, rownames=['True values'], colnames=['k-NN']), '\n')
