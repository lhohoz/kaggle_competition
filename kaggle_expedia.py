
# coding: utf-8

# In[67]:

import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
from sklearn.decomposition import PCA
#import seaborn as sns
#import ml_metrics as metrics

df_train = pd.read_csv('train_expedia.csv', chunksize=10000000, usecols=['srch_destination_id', 
                                                                         'orig_destination_distance',
                                                                         'is_booking',
                                                                         'hotel_cluster'])
df_test = pd.read_csv('test_expedia.csv', usecols=['srch_destination_id',
                                                   'orig_destination_distance'])
dest = pd.read_csv('destinations_expedia.csv')
dest_cv = dest.drop(['srch_destination_id'], axis=1)

#统计距离空值的用户
#distant_nan = df_train['user_id'][pd.isnull(df_train.orig_destination_distance)].value_counts()
#distant_nan.plot(kind='bar',colormap="Set2",figsize=(15,5))

#按年生成时间序列图
#df_train['srch_ci'] = df_train['srch_ci'].apply(lambda x: (str(x)[:7]) if x == x else np.nan)
#agg = df_train.groupby(['srch_ci'])['is_booking'].sum()
#ax1 = agg.plot(legend=True,marker='o',title="Total Bookings", figsize=(15,5)) 

#距离分布直方图
#distance = df_train['orig_destination_distance'][df_train.is_booking == 1]
#distance.plot(kind='hist',colormap="Set1", title="booking_distance")

def top_five(group):
    distance = group['detail'].values
    hotel_group = group['hotel_cluster'].values
    sort_group = hotel_group[np.argsort(distance)[::-1]][:5]
    return np.array_str(sort_group)[1:-1]

def prob(x):
    prob = np.exp(x) / np.exp(x).sum()
    return prob.max()

train = []
train_dis = []
most_pop = []
#train_group = df_train.groupby(['srch_destination_id', 
#                                'hotel_cluster'])['is_booking'].agg(['sum', 'count']).reset_index()
#train_group_dis = df_train.groupby(['orig_destination_distance', 
#                                    'hotel_cluster'])['is_booking'].agg(['sum', 'count']).reset_index()

for chunk in df_train:    
    train_group = chunk.groupby(['srch_destination_id', 
                                 'hotel_cluster'])['is_booking'].agg(['sum', 'count']).reset_index()
    
    train_group_dis = chunk.groupby(['orig_destination_distance', 
                                     'hotel_cluster'])['is_booking'].agg(['sum', 'count']).reset_index()  
    
    most_pop_all = chunk.groupby('hotel_cluster')['is_booking'].sum().nlargest(5).index
    
    train.append(train_group)
    train_dis.append(train_group_dis)
    most_pop.append(most_pop_all)
    
train = pd.concat(train, axis=0)
train_dis = pd.concat(train_dis, axis=0)

#2次聚合
train_group = train.groupby(['srch_destination_id', 
                             'hotel_cluster']).sum().reset_index()

train_group_dis = train_dis.groupby(['orig_destination_distance', 
                                     'hotel_cluster']).sum().reset_index()

#destinations特征处理
dest_cv = dest_cv.apply(prob, axis=1)
dest = pd.concat((dest['srch_destination_id'], dest_cv), axis=1)
train_group = pd.merge(train_group, dest, how='left', on=['srch_destination_id'])
m_prob = train_group[0].min()
train_group[0] = train_group[0].apply(lambda x: m_prob if pd.isnull(x) else x)
#pca = PCA(n_components=1)
#dest_small = pca.fit_transform(dest[['d{0}'.format(i + 1) for i in range(149)]])
#dest_small = pd.DataFrame(dest_small)
#dest_small['srch_destination_id'] = dest['srch_destination_id']
#train_group = pd.merge(train_group, dest_small, how='left', on=['srch_destination_id'])
#m_prob = train_group[0].min()
#train_group[0] = train_group[0].apply(lambda x: m_prob if pd.isnull(x) else x)


#0.30334(old) = x['sum'] + x[0] + 0.05 * x['count']
#0.30340(old) = 0.95 * x['sum'] + x[0] + 0.05 * x['count']
#0.30342(old) = 1.5 * x['sum'] + x[0] + 0.15 * x['count']
#0.30343(old) = 1.5 * x['sum'] + 8 * x[0] + 0.15 * x['count']
#0.38879(new) = 0.8456 * x['sum'] + (1 - 0.8456) * x['count']
#0.45778(new) = 1.5 * x['sum'] + x[0] + 0.15 * x['count']
#0.45779(new) = 1.5 * x['sum'] + 8 * x[0] + 0.15 * x['count']
#0.45782(new) = 1.7 * x['sum'] + 0.05 * x[0] + 0.18 * x['count']
train_group['detail'] = train_group.apply(lambda x: (1.7 * x['sum'] + 0.05 * x[0] + 0.18 * x['count']), axis=1)
train_group_dis['detail'] = train_group_dis.apply(lambda x: (1.7 * x['sum'] + 0.18 * x['count']), axis=1)

#id聚合
train_resault = train_group.groupby(['srch_destination_id']).apply(top_five).reset_index()
train_resault_dis = train_group_dis.groupby(['orig_destination_distance']).apply(top_five).reset_index()

#test_merge
df_test_id = pd.merge(df_test, train_resault, how='left', on=['srch_destination_id'])
df_test_dis = pd.merge(df_test, train_resault_dis, how='left', on=['orig_destination_distance'])
df_test_dis[0][df_test_dis[0].isnull()] = df_test_id[0][df_test_dis[0].isnull()]

#提取频繁集并替换空值
most_pop_all = np.array_str(most_pop[0])[1:-1]
df_test_dis[0].fillna(most_pop_all, inplace=True)

#提交结果
sub = df_test_dis.rename(columns={0: 'hotel_cluster'})
sub['hotel_cluster'].to_csv('sub_expedia.csv', header=True, index_label='id')
