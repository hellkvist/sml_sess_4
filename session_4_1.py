import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb

plt.style.use('seaborn-white')


biopsy = pd.read_csv('../Data/biopsy.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()

np.random.seed(1)
n_train = round(len(biopsy)/2)
train_index = np.random.choice(len(biopsy), size=n_train, replace=False)
train_index_bool = biopsy.index.isin(train_index)

train = biopsy.iloc[train_index_bool]
test = biopsy.iloc[~train_index_bool]

cols_in = ['V3', 'V4', 'V5']
cols_out = ['class']
X = train[cols_in]
y = train[cols_out].values.flatten()


### logistic regression
reg = skl_lm.LogisticRegression().fit(X, y)
y_hat = reg.predict(test[cols_in])
y_true = test[cols_out].values.flatten()
correct_bool = (y_hat == y_true)
n_correct = correct_bool.sum()
print('Log. Reg:\nNo. of correct classifications: ', n_correct, ' out of ', n_train)

crosstab_df = pd.crosstab(y_true, y_hat, rownames=['true'], colnames=['prediction'])
print(crosstab_df)

n_positives = len(y_true[y_true == 'malignant'])
y_true_bool = y_true == 'benign'
y_probs = reg.predict_proba(test[cols_in])[:, 0]
n_r = 99
r_vec = np.array(range(1, n_r+1))/(n_r+1)
y_hat = np.empty(shape=(n_r, n_train-1), dtype=bool)
TPR = np.zeros(shape=(n_r,))
FPR = np.zeros(shape=(n_r,))
for ri in range(n_r):
    r = r_vec[ri]
    for yi in range(len(y_probs)):
        y_temp = y_probs[yi]
        if y_temp > r:  # below r malignant=1, above r benign=0
            y_hat[ri, yi] = 0
        else:
            y_hat[ri, yi] = 1
    crosstab_df = pd.crosstab(y_true_bool, y_hat[ri, :])
    if crosstab_df.values.shape == (2,2):

        FPR[ri] = crosstab_df.values[0, 1]/n_positives
        TPR[ri] = crosstab_df.values[1, 1]/n_positives


plt.plot(FPR, TPR, '--o')
plt.show()



#######         LDA         ########
lda = skl_da.LinearDiscriminantAnalysis().fit(X, y)
y_hat = lda.predict(test[cols_in])
y_true = test[cols_out].values.flatten()
correct_bool = (y_hat == y_true)
n_correct = correct_bool.sum()
print('\nLDA:\nNo. of correct classifications: ', n_correct, ' out of ', n_train)

crosstab_df = pd.crosstab(y_true, y_hat, rownames=['true'], colnames=['prediction'])
print(crosstab_df)

#######         QDA         ########
qda = skl_da.LinearDiscriminantAnalysis().fit(X, y)
y_hat = qda.predict(test[cols_in])
y_true = test[cols_out].values.flatten()
correct_bool = (y_hat == y_true)
n_correct = correct_bool.sum()
print('\nQDA:\nNo. of correct classifications: ', n_correct, ' out of ', n_train)

crosstab_df = pd.crosstab(y_true, y_hat, rownames=['true'], colnames=['prediction'])
print(crosstab_df)

#####¤¤         kNN         #######
knn = skl_nb.KNeighborsClassifier(n_neighbors=1).fit(X, y)
y_hat = knn.predict(test[cols_in])
y_true = test[cols_out].values.flatten()
correct_bool = (y_hat == y_true)
n_correct = correct_bool.sum()
print('\nKNN k=1:\nNo. of correct classifications: ', n_correct, ' out of ', n_train)

crosstab_df = pd.crosstab(y_true, y_hat, rownames=['true'], colnames=['prediction'])
print(crosstab_df)

### k = 50
knn = skl_nb.KNeighborsClassifier(n_neighbors=5).fit(X, y)
y_hat = knn.predict(test[cols_in])
y_true = test[cols_out].values.flatten()
correct_bool = (y_hat == y_true)
n_correct = correct_bool.sum()
print('\nKNN k=5:\nNo. of correct classifications: ', n_correct, ' out of ', n_train)

crosstab_df = pd.crosstab(y_true, y_hat, rownames=['true'], colnames=['prediction'])
print(crosstab_df)