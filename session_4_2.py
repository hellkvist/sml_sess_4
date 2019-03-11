import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb

np.random.seed(2)
N = 100
x1 = np.random.uniform(0, 10, N)
x2 = np.random.uniform(0, 10, N)
y = np.repeat(1, N)
y[x1<4] = 0
y[x2<4] = 0
X = pd.DataFrame({'x1': x1, 'x2': x2, 'x1*x2': x1*x2})

# learn a logistic regression model
model = skl_lm.LinearRegression()
model.fit(X, y)

# classify the points in the whole domain
res = 0.1   # resolution of the squares
xs1 = np.arange(0, 10.1, 0.1)
xs2 = np.arange(0, 10.1, 0.1)
xs1, xs2 = np.meshgrid(xs1, xs2)    # Creating the grid for all the data points
X_all = pd.DataFrame({'x1': xs1.flatten(), 'x2': xs2.flatten(), 'x1*x2': (xs1*xs2).flatten()})
prediction = model.predict(X_all)

plt.figure(figsize=(10, 5))

# Plot of the prediction for all the points in the space
colors = np.where(prediction==0,'skyblue','lightsalmon')
plt.scatter(xs1, xs2, s = 90, marker='s', c=colors)

# Plot of the data points and their label
color = np.where(y==0, 'b', 'r')
plt.scatter(x1, x2, c=color)

plt.title('Logistic regression decision boundary')
plt.show()