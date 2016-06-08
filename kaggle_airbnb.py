
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


def train_test_shape():
    df_train = pd.read_csv('train_users_2.csv')
    df_test = pd.read_csv('test_users.csv')
    label = df_train['country_destination']
    id_list = df_test['id']
    return df_train.shape[0], df_test.shape[0], label, id_list


def data_set():
    df_train = pd.read_csv('train_users_2.csv')
    df_test = pd.read_csv('test_users.csv')
    df_train = df_train.drop('country_destination', axis=1)
    df = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    return df


def sessions_stats(group):
    group.fillna(0, inplace=True)

    if group.count() == 0:
        return {'sessions_total_duration': group.max() - group.min(),
                'average_action_duration': 0,
                'actions_total_count': 0}
    else:
        return {'sessions_total_duration': group.max() - group.min(),
                'average_action_duration': (group.max() - group.min()) / group.count(),
                'actions_total_count': group.count()}

    
def session():
    sessions_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    df = pd.read_csv('sessions.csv')
    
    #对数值变量进行描述性统计,groupby后连接apply函数，可自行编写统计规则
    df_stats = df['secs_elapsed'].groupby(df['user_id']).apply(sessions_stats).unstack()
    df_stats['actions_total_count'] = df_stats['actions_total_count'].apply(
                                        lambda x: np.sqrt(x/3600))
    df_stats['average_action_duration'] = df_stats['average_action_duration'].apply(
                                            lambda x: np.sqrt(x/3600))
    
    normalize_feats = ['actions_total_count',
                       'average_action_duration', 
                       'sessions_total_duration']
    
    #将属性缩放到一定范围，针对方差很小的数值，稀疏矩阵为0的条目
    for f in normalize_feats:
        df_stats[f] = sessions_scaler.fit_transform(
                      df_stats[f].reshape(-1, 1)).astype(int)
        
    df_sactions = df.groupby(['user_id', 'action_detail', 'action_type'], 
                             as_index=False).count()
    
    df_sactions.drop(['secs_elapsed', 'action', 'device_type'], 
                     axis=1, inplace=True)
    
    ohe_features = ['action_detail', 'action_type']
    for f in ohe_features:
        df_dummy = pd.get_dummies(df_sactions[f], prefix=f)
        df_sactions.drop([f], axis=1, inplace = True)
        df_sactions = pd.concat((df_sactions, df_dummy.astype(int)), axis=1)
        
    #将复数条用户数据groupby成1条记录
    df_sactions = df_sactions.groupby(['user_id']).sum().reset_index()
    
    df_joined = df_sactions.join(df_stats, on=['user_id'], how='left')
    df_joined.rename(columns={'user_id': 'id'}, inplace=True)
    
    return df_joined


def transform_data(df):
    df['date_account_created'] = pd.to_datetime(df['date_account_created'])
    df['date_first_booking'] = pd.to_datetime(df['date_first_booking'])
    df['timestamp_first_active'] = pd.to_datetime(df['timestamp_first_active']) 
    
    df['timestamp_first_active_date'] = df['timestamp_first_active'].apply(
                                          lambda x: x.strftime('%Y-%m-%d'))
    df['timestamp_first_active_date'] = pd.to_datetime(df['timestamp_first_active_date'])  
    
    df['toacc_date_range'] = (df['date_account_created'] - 
                              df['timestamp_first_active_date'])\
                             / np.timedelta64(1, 'ns')    
    df['tobooking_date_range'] = (df['date_first_booking'] - 
                                  df['date_account_created'])\
                             / np.timedelta64(1, 'ns') 
    df['tobooking_date_range'].fillna(0, inplace = True)
    
    if np.where(df['tobooking_date_range'] < 0):
        df['tobooking_date_range'].values[np.where(df['tobooking_date_range'] < 0)] = 0
        
    df = df.drop(['date_account_created', 
                  'timestamp_first_active', 'date_first_booking',
                  'timestamp_first_active_date'], axis=1)
    
    df['age'].fillna(-1, inplace = True)
    if np.where(df['age'] > 100):
        df['age'].values[np.where(df['age'] > 100)] = -1
    if np.where(df['age'] < 20):
        df['age'].values[np.where(df['age'] < 20)] = -1
            
    ohe_feats = ['gender', 'signup_method', 
                 'signup_flow', 'language', 
                 'affiliate_channel', 'affiliate_provider', 
                 'first_affiliate_tracked', 
                 'signup_app', 'first_device_type', 
                 'first_browser']
    
    for f in ohe_feats:
        df_dummy = pd.get_dummies(df[f], prefix=f)
        df = df.drop([f], axis=1)
        df = pd.concat((df, df_dummy), axis=1)  
    return df


def model(x_train, x_test, y_train, id_test):
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, 
                                 random_state=1) 
    clf.fit(x_train, y_train)
    y_predict = clf.predict_proba(x_test) 
    #sample_submission = {}
    #sample_submission['country'] = le.inverse_transform(y_predict)
    #sample_submission['id'] = id_test
    #sub = pd.DataFrame(sample_submission, columns=['id', 'country'])
    #return sub.to_csv('sub.csv',index=False)
    ids = []
    cts = []
    for i in range(len(id_test)):
        idx = id_test[i]
        ids += [idx] * 5
        cts += le.inverse_transform(np.argsort(y_predict[i])[::-1])[:5].tolist()
    sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
    return sub.to_csv('sub_RandomForest.csv',index=False)


x = transform_data(data_set())
x_all = pd.merge(x, session(), how='left', on='id') 
x_session = x_all.drop('id', axis=1)
x_session.fillna(0, inplace=True)

x_train = x_session[:train_test_shape()[0]]
x_test = x_session[train_test_shape()[0]:]

le = LabelEncoder()
y_train = le.fit_transform(train_test_shape()[2].values)
id_test = train_test_shape()[3]

model(x_train, x_test, y_train, id_test)
