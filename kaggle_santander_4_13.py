
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import xgboost as xgb


df_train = pd.read_csv('train.csv')#,iterator = True)
#df_train = f_train.get_chunk(100)
df_test = pd.read_csv('test.csv')#,iterator = True)
#df_test = f_test.get_chunk(100)

constant_col = [v for i, v in enumerate(df_train.columns)
                    if len(pd.unique(df_train[df_train.columns[i]])) == 1]
df_train.drop(constant_col, axis=1, inplace=True)
df_test.drop(constant_col, axis=1, inplace=True)

same_col = []
i = 0
while True:
    if i == len(df_train.columns) - 1:
        break
    start_col = df_train[df_train.columns[i]].values
    for j in range(i + 1, len(df_train.columns)):
        next_col = df_train[df_train.columns[j]].values
        if list(start_col) == list(next_col):
            same_col.append(df_train.columns[j])
    i += 1

df_train.drop(same_col, axis=1, inplace=True)
df_test.drop(same_col, axis=1, inplace=True)
    
x_train = df_train.drop(['ID', 'TARGET'], axis=1)
x_test = df_test.drop(['ID'], axis=1)
y_train = df_train.values[:,-1]
id_test = df_test['ID']
    
def var_var(x):
    return x[list(np.where(x != 0)[0])].var()
    
x_train['var_var'] = x_train.apply(var_var, axis=1)
x_test['var_var'] = x_test.apply(var_var, axis=1)
x_train['zero'] = (x_train == 0).astype(int).sum(axis=1)
x_test['zero'] = (x_test == 0).astype(int).sum(axis=1)

#x_train['var3'] = x_train['var3'].replace(-999999,2)
#x_test['var3'] = x_test['var3'].replace(-999999,2)
#x_train['var38'] = np.log(x_train['var38'])
#x_test['var38'] = np.log(x_test['var38'])

xgb_train = xgb.DMatrix(x_train, label=y_train)
xgb_test = xgb.DMatrix(x_test)

watchlist = [(xgb_train, 'train')]
num_round = 285  #500(0.837661)|400(0.838669)|400(0.839560)|310(0.839809)|300(0.840096)|285(0.840128)

param = {}
param['objective'] = 'binary:logistic'
param['eval_metric'] = 'auc'
param['subsample'] = 0.75  #0.95(0.838669)|0.85(0.839560)|0.95(0.839809)|0.75(0.840096)
param['colsample_bytree'] = 0.65 #0.85(0.838669)|0.87(0.839560)|0.85(0.839809)|0.65(0.840096)
param['eta'] = 0.03     #0.02(0.838669)|0.019(0.839560)|0.03(0.839809)|0.03(0.840096)
param['max_depth'] = 5  #7(0.838669)|6(0.839560)|5(0.839809)|5(0.840096)
param['silent'] = 1    #1(0.838669)|1(0.839560)
param['nthread'] = 4   #5(0.838669)|5(0.839560)|4(0.839809)
param['verbose'] = 1   #2(0.838669)|1(0.839560)
param['booster'] = 'gbtree'
param['maximise'] = False

clf = xgb.train(param, xgb_train, num_round, watchlist)
y_pred = clf.predict(xgb_test)

sample_submission = {}
sample_submission['ID'] = id_test
sample_submission['TARGET'] = y_pred
sub = pd.DataFrame(sample_submission, columns=['ID', 'TARGET'])
sub.to_csv('sub_santander.csv',index=False)
