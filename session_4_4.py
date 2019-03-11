import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from sklearn.preprocessing import StandardScaler

np.random.seed(1)
x1 = np.random.normal(0, 1, size=(200,))
x2 = np.random.normal(0, 100, size=(200,))

y = np.where(x1*x2 > 0, 1, 0)

X_train = np.column_stack((x1[0:100], x2[0:100]))
scaler_X = StandardScaler().fit(X_train)
X_train = scaler_X.transform(X_train)
y_train = y[0:100]

X_test = np.column_stack((x1[100:200], x2[100:200]))
X_test = scaler_X.transform(X_test)
y_test = y[100:200]

knn = skl_nb.KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
y_hat = knn.predict(X_test)

xs1 = np.arange(min(X_train[:, 0]), max(X_train[:, 0]), 0.1)
xs2 = np.arange(min(X_train[:, 1]), max(X_train[:, 1]), 0.1)
xs1, xs2 = np.meshgrid(xs1, xs2)
Xs = np.column_stack((xs1.flatten(), xs2.flatten()))
boundary = knn.predict(Xs)
colors = np.where(boundary == 1, 'lightsalmon', 'skyblue')
plt.scatter(scaler_X.inverse_transform(Xs)[:, 0], scaler_X.inverse_transform(Xs)[:, 1], marker='s', s=40, c=colors)


colors = np.where(y_test == 1, 'r', 'b')
plt.scatter(x1[100:200], x2[100:200], c=colors)

plt.show()