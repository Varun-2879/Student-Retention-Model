import pandas as pd
import numpy as np

#Import data, not shown due to security reasons

print("the size of data before dropping duplicates:", dataset.size)

dataset.drop_duplicates(keep = 'first', inplace = True) 
print("the size of data after dropping duplicates:", dataset.size)

dataset['WD_date'] = dataset['WD_date'].astype(str)
dataset['Withdraw_YN'] = np.where(dataset['WD_date'] == 'nan', 0, 1)

dataset['PeopleOrgId'] = dataset['PeopleOrgId'].str.replace("P", "")

date_diff =  dataset.Decision_Stage_date - dataset.Application_Recvd_date
dataset['date_diff'] = date_diff

d1 = dataset[['PeopleOrgId', 'Admit_AcademicYear', 'Withdraw_YN', 'Direct_miles', 'CreditsEnrolled', 'date_diff']]


d1['date_diff'] = d1['date_diff'].values.astype(float) 
d1['Admit_AcademicYear'] = d1['Admit_AcademicYear'].values.astype(float) 
d1['Direct_miles'] = d1['Direct_miles'].values.astype(float) 
d1['CreditsEnrolled'] = d1['CreditsEnrolled'].values.astype(float) 
d1['Withdraw_YN'] = d1['Withdraw_YN'].values.astype(float) 
d1['PeopleOrgId'] = d1['PeopleOrgId'].values.astype(float) 


from sklearn.model_selection import train_test_split

X = d1.drop('Withdraw_YN', axis = 1)
Y = d1['Withdraw_YN']


print("Shape of X:",X.shape)
print("Shape of Y:",Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y,stratify = Y, test_size = 0.33, random_state = 40)

print("Shape of X_train:",X_train.shape)
print("Shape of y_train:",y_train.shape)
print("Shape of X_test:",X_test.shape)
print("Shape of y_test:",y_test.shape)

print("Null values in X_train: \n",X_train.isna().sum())
print("Null values in X_test: \n",X_test.isna().sum())

print('--------')

print("Null values in y_train: \n",y_train.isna().sum())
print("Null values in y_test: \n",y_test.isna().sum())


mean_Admit_AcademicYear_X_train = X_train['Admit_AcademicYear'].mean()
X_train['Admit_AcademicYear'].fillna(value = mean_Admit_AcademicYear_X_train, inplace = True)
mean_Admit_AcademicYear_X_test = X_test['Admit_AcademicYear'].mean()
X_test['Admit_AcademicYear'].fillna(value = mean_Admit_AcademicYear_X_test, inplace = True)

print("Null values in X_train: \n",X_train.isna().sum())
print("Null values in X_test: \n",X_test.isna().sum())

print('--------')

print("Null values in y_train: \n",y_train.isna().sum())
print("Null values in y_test: \n",y_test.isna().sum())

from sklearn import preprocessing

X_train_scaler = preprocessing.StandardScaler().fit(X_train)
X_train_std = X_train_scaler.transform(X_train)

X_test_scaler = preprocessing.StandardScaler().fit(X_test)
X_test_std = X_test_scaler.transform(X_test)


Y_TRAIN = y_train
Y_TEST = y_test


Y_train_std = Y_TRAIN
Y_test_std = Y_TEST

print( 'shape of X_TRAIN = ', (X_train_std).shape)
print( 'shape of X_TEST = ',(X_test_std).shape)
print( 'shape of Y_TRAIN = ',(Y_train_std).shape)
print( 'shape of Y_TEST = ',(Y_test_std).shape)


from imblearn.over_sampling import SMOTE


print("Before OverSampling, counts of label '1': {}".format(sum(Y_train_std==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(Y_train_std==0)))

sm = SMOTE(random_state=4)
X_train_std, Y_train_std = sm.fit_resample(X_train_std, Y_train_std.ravel())

print('After OverSampling, the shape of X_train: {}'.format(X_train_std.shape))
print('After OverSampling, the shape of Y_train: {} \n'.format(Y_train_std.shape))

print("After OverSampling, counts of label '1': {}".format(sum(Y_train_std==1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_std==0)))

d2 = dataset.drop(dataset.columns[[0,1,12,13,14,17,20,22,21,19]], axis=1)

print('----------------------------------------------------------')

print(d1.shape)
print(d2.shape)

d1.drop("Withdraw_YN", axis=1, inplace=True)

print('----------------------------------------------------------')

d2 = pd.concat([d1, d2], axis = 1)

X_final = d2
Y_final = Y

print("The final merged independent variable X: \n", X_final)
print("The final merged dependent variable Y: \n", Y_final)

X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final, Y_final, stratify=Y_final, test_size = 0.40, random_state = 42)



mean_Admit_AcademicYear_X_train = X_train_final['Admit_AcademicYear'].mean()
X_train_final['Admit_AcademicYear'].fillna(value = mean_Admit_AcademicYear_X_train, inplace = True)
mean_Admit_AcademicYear_X_test = X_test_final['Admit_AcademicYear'].mean()
X_test_final['Admit_AcademicYear'].fillna(value = mean_Admit_AcademicYear_X_test, inplace = True)


X_train_final['Admit_AcademicTerm'].fillna(value = 'FALL', inplace = True)
X_test_final['Admit_AcademicTerm'].fillna(value = 'FALL', inplace = True)


X_train_final['Ethnicity'].fillna(value = 'WNH', inplace = True)
X_test_final['Ethnicity'].fillna(value = 'WNH', inplace = True)


X_train_final['MaritalStatus'].fillna(value = 'UNKN', inplace = True)
X_test_final['MaritalStatus'].fillna(value = 'UNKN', inplace = True)


X_train_final['Admit_as'].fillna(value = 'F', inplace = True)
X_test_final['Admit_as'].fillna(value = 'F', inplace = True)



X_train_final['AcademicInterest'].fillna(value = 'UND', inplace = True)
X_test_final['AcademicInterest'].fillna(value = 'UND', inplace = True)

X_train_final['InternationalStudent'].fillna(value = 'N', inplace = True)
X_test_final['InternationalStudent'].fillna(value = 'N', inplace = True)


X_train_final['EPS_code'].fillna(value = 'NH-3', inplace = True)
X_test_final['EPS_code'].fillna(value = 'NH-3', inplace = True)


X_train_final['Residence'].fillna(value = 'R', inplace = True)
X_test_final['Residence'].fillna(value = 'R', inplace = True)


X_train_final['FPL'].fillna(value = 'F', inplace = True)
X_test_final['FPL'].fillna(value = 'P', inplace = True)

print(X_train_final.isna().sum())
print(X_test_final.isna().sum())

print(X_train_final.shape)
print(X_test_final.shape)

print(y_train_final.shape)

print(y_test_final.shape)


print(X_train.isna().sum())
print(X_test.isna().sum())

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

X_train_final = pd.get_dummies(X_train_final)
X_test_final = pd.get_dummies(X_test_final)

X_train_trial, X_test_trial = X_train_final.align(X_test_final, join = 'inner', axis = 1)

print(X_train_trial.head())
print(X_test_trial.head())

print(X_train_trial.shape)
print(X_test_trial.shape)

print(y_train_final.shape)
print(y_test_final.shape)

from imblearn.over_sampling import SMOTE

print(y_train_final.shape)

print(y_test_final.shape)


print("Before OverSampling, counts of label '1': {}".format(sum(y_train_final==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train_final==0)))

sm = SMOTE(random_state=2)
X_train_trial, y_train_final = sm.fit_resample(X_train_trial, y_train_final.ravel())

print('After OverSampling, the shape of X_train: {}'.format(X_train_trial.shape))
print('After OverSampling, the shape of Y_train: {} \n'.format(y_train_final.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_final==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_final==0)))

import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score


print(X_train_trial.shape)
print(X_test_trial.shape)
print(y_train_final.shape)
print(y_test_final.shape)

params = {
 'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
 'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
 'min_child_weight' : [ 1, 3, 5, 7 ],
 'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ]
}


clf =  XGBClassifier()

rs_model=RandomizedSearchCV(clf,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

rs_model.fit(X_train_trial, y_train_final)

rs_model.best_estimator_

y_predicted = rs_model.predict(X_test_trial)


print(roc_auc_score(y_test_final, y_predicted))


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits = 10, random_state = 1, shuffle = True)

scores = cross_val_score(clf,X_train_trial, y_train_final ,cv=cv, n_jobs = -1)

print(scores)

