# -*- coding: utf-8 -*-
"""
@author: Yetta
"""

import gc
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

#load data
data_path='./'
rating = pd.read_csv(data_path+'data/train/rating.txt',dtype=int,sep=' ',header=None,names=['uid','mid','rating'])
bigtag = pd.read_csv(data_path+'data/train/bigtag.txt',dtype=int,sep=' ',header=None,names=['uid','mid','tid'])
movie = pd.read_csv(data_path+'data/train/movie.txt',dtype=int,sep=' ',header=None,names=['mid','tag1','tag2','tag3','tag4','tag5','tag6','tag7','tag8'])
train = pd.read_csv(data_path+'data/valid/validation.txt',dtype=int,sep=' ',header=None,names=['uid','tid','y'])
test = pd.read_csv(data_path+'data/test/test_phase2.txt',dtype=int,sep=' ',header=None,names=['uid','tid'])

#FE
def get_utid(df,uid='uid',tid='tid'):
    df['uid_tid'] = df[uid].astype(str)+"_"+df[tid].astype(str)
    
get_utid(train)
get_utid(test)
get_utid(bigtag)

"merge rating&movie"
mov_rat = pd.merge(rating,movie,on='mid')
mov_rat_f = pd.concat([mov_rat[['uid','mid','rating',f'tag{i}']].rename(columns={f'tag{i}':'tid'}) for i in range(1,9)],ignore_index=True)
get_utid(mov_rat_f)

train['set'] = 'train'
test['set'] = 'test'
dff = pd.concat([train,test],ignore_index=True)

def fe_rating_gp_utid_stat(df):
    "rating&movie: rating groupby uid_tid stat"
    #total stat
    stat=['count','nunique','max','mean','median','min']
    res0 = mov_rat_f.groupby(['uid_tid'])['rating'].agg(stat)
    for s in stat:
        df[f'rating_gp_utid_{s}'] = df['uid_tid'].map(res0[s])
    #split tid stat
    for t in [f'tag{i}' for i in range(1,9)]:
        get_utid(mov_rat,uid='uid',tid=t)
        res0 = mov_rat.groupby(['uid_tid'])['rating'].agg(stat)
        for s in stat:
            df[f'rating_gp_u{t}_{s}'] = df['uid_tid'].map(res0[s])
        
fe_rating_gp_utid_stat(dff)

#get mid-tid pairs
mov_tag = pd.concat([movie[['mid',f'tag{i}']].rename(columns={f'tag{i}':'tid'}) for i in range(1,9)],ignore_index=True)

def fe_tid_freq(df):
    "tag freq"
    #all movie tag freq 
    df['tid_freq'] = df['tid'].map(mov_tag.tid.value_counts())
    #split movie tag freq
    for c in ['tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'tag6', 'tag7', 'tag8']:
        df[f'{c}_freq'] = df['tid'].map(movie[c].value_counts())

fe_tid_freq(dff)

def tag_fe(df,bigtag):
    "bigtag fe"
    bigtag_n1 = bigtag[bigtag.tid==-1]
    bigtag = bigtag[bigtag.tid>-1]
    get_utid(bigtag)
    
    # labeled tid stat
    stat = ['count','nunique']
    res0 = bigtag.groupby('tid')['uid'].agg(stat)
    res1 = bigtag.groupby('tid')['mid'].agg(stat)
    res2 = bigtag.groupby('uid_tid')['mid'].agg(stat)
    for i in stat:
        df[f'bigtag_uid_gp_tid_{i}'] = df['tid'].map(res0[i])
        df[f'bigtag_mid_gp_tid_{i}'] = df['tid'].map(res1[i])
        df[f'bigtag_mid_gp_utid_{i}'] = df['uid_tid'].map(res2[i])    
    for c in ['uid_gp_tid','mid_gp_tid','mid_gp_utid']:
        df[f'bigtag_{c}_rate'] = df[f'bigtag_{c}_nunique']/df[f'bigtag_{c}_count']
        df[f'bigtag_{c}_diff'] = df[f'bigtag_{c}_nunique']-df[f'bigtag_{c}_count']
    
    #graph uid-tid-mid-tid,like stat
    mov_tag1 = pd.merge(bigtag,mov_tag,on='mid')
    get_utid(mov_tag1,'uid','tid_y')
    like_uid_tid = mov_tag1.groupby('uid_tid')['mid'].agg(['count','nunique'])
    for s in like_uid_tid:
        df[f'graph_utmt_like_mid_gp_utid_{s}'] = df['uid_tid'].map(like_uid_tid[s])
    df['graph_utmt_like_mid_gp_utid_diff'] = df['graph_utmt_like_mid_gp_utid_nunique']-df['graph_utmt_like_mid_gp_utid_count']
    df['graph_utmt_like_mid_gp_utid_rate'] = df['graph_utmt_like_mid_gp_utid_nunique']/df['graph_utmt_like_mid_gp_utid_count']
    df['graph_utmt_like_tid_freq'] = df['tid'].map(mov_tag1.tid_y.value_counts(normalize=True)).fillna(-0.01)
    # dislike stat
    mov_tag2 = pd.merge(bigtag_n1,mov_tag,on='mid')
    get_utid(mov_tag2,'uid','tid_y')
    hate_uid_tid = mov_tag2.groupby('uid_tid')['mid'].agg(['count','nunique'])
    for s in hate_uid_tid:
        df[f'graph_utmt_dislike_mid_gp_utid_{s}'] = df['uid_tid'].map(hate_uid_tid[s])
    df['graph_utmt_dislike_mid_gp_utid_diff'] = df['graph_utmt_dislike_mid_gp_utid_nunique']-df['graph_utmt_dislike_mid_gp_utid_count']
    df['graph_utmt_dislike_mid_gp_utid_rate'] = df['graph_utmt_dislike_mid_gp_utid_nunique']/df['graph_utmt_dislike_mid_gp_utid_count']
    df['graph_utmt_dislike_tid_freq'] = df['tid'].map(mov_tag2.tid_y.value_counts(normalize=True)).fillna(-0.01)
    
    # graph uid-tid-uid-tid
    utut = pd.merge(bigtag[['uid','tid']],bigtag[['uid','tid']],on='tid')
    utut = utut[utut.uid_x!=utut.uid_y]
    del utut['tid'];gc.collect()
    utut = pd.merge(utut,bigtag[['uid','tid']],left_on='uid_y',right_on='uid')
    del utut['uid_y'];gc.collect()
    utut['uid_tid'] = utut['uid_x'].astype(str)+"_"+utut['tid'].astype(str)
    res = utut.groupby('uid_tid')['uid'].agg(['count','nunique'])
    for s in res:
        df[f'graph_utut_{s}'] = df['uid_tid'].map(res[s])
    df['graph_utut_nunique_rate'] = df['graph_utut_nunique']/df['graph_utut_count']
    df['graph_utut_nunique_diff'] = df['graph_utut_nunique']-df['graph_utut_count']
    df['graph_utut_freq']= df['tid'].map(utut.tid.value_counts(normalize=True)).fillna(-0.01)
    del utut;gc.collect()

tag_fe(dff,bigtag)

def get_lgb(train_x0,test_x0,xgb_cols,params,y='dlBw',esr=20):
    'train lgb model'
    lgb_train0 = lgb.Dataset(train_x0[xgb_cols].values,label=train_x0[y])
    lgb_test0 = lgb.Dataset(test_x0[xgb_cols].values,label=test_x0[y])
    res0 = {}
    gbm0 = lgb.train(params,
                     lgb_train0,
                     valid_sets=[lgb_train0,lgb_test0],
                     verbose_eval=500,
                     num_boost_round=2000,
                     early_stopping_rounds=esr,
                     evals_result=res0)
    return res0,gbm0

def kflod_gbm(train,use_cols,params,k=5,y='dlBw',esr=20,random_state=666):   
    'get cv lgb model'
    train_idx_dt = {}
    test_idx_dt = {}
    tres = {}
    tgbm = {}
    skf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    for index, (train_index, test_index) in enumerate(skf.split(train,train[y])):   
        train_idx_dt[index] = train.iloc[train_index]
        test_idx_dt[index] = train.iloc[test_index]   
        tres[index],tgbm[index] = get_lgb(train_idx_dt[index],test_idx_dt[index],use_cols,params,y,esr)
        test_idx_dt[index]['pred'] = tgbm[index].predict(test_idx_dt[index][use_cols])
    score_mean = [x['valid_1']['auc'][-esr-1] for _,x in tres.items()]
    return tres,tgbm,test_idx_dt,score_mean

use_cols=['tid', 'bigtag_uid_gp_tid_count', 'bigtag_uid_gp_tid_nunique', 'bigtag_uid_gp_tid_rate', 'bigtag_uid_gp_tid_diff', 'bigtag_mid_gp_tid_count', 'bigtag_mid_gp_utid_count', 'bigtag_mid_gp_tid_nunique', 'bigtag_mid_gp_utid_nunique', 'bigtag_mid_gp_tid_rate', 'bigtag_mid_gp_tid_diff', 'bigtag_mid_gp_utid_rate', 'bigtag_mid_gp_utid_diff', 'tid_freq', 'tag1_freq', 'tag2_freq', 'tag3_freq', 'tag4_freq', 'tag5_freq', 'tag6_freq', 'tag7_freq', 'tag8_freq', 'rating_gp_utid_count', 'rating_gp_utid_nunique', 'rating_gp_utid_max', 'rating_gp_utid_mean', 'rating_gp_utid_median', 'rating_gp_utid_min', 'rating_gp_utag1_count', 'rating_gp_utag1_nunique', 'rating_gp_utag1_max', 'rating_gp_utag1_mean', 'rating_gp_utag1_median', 'rating_gp_utag1_min', 'rating_gp_utag2_count', 'rating_gp_utag2_nunique', 'rating_gp_utag2_max', 'rating_gp_utag2_mean', 'rating_gp_utag2_median', 'rating_gp_utag2_min', 'rating_gp_utag3_count', 'rating_gp_utag3_nunique', 'rating_gp_utag3_max', 'rating_gp_utag3_mean', 'rating_gp_utag3_median', 'rating_gp_utag3_min', 'rating_gp_utag4_count', 'rating_gp_utag4_nunique', 'rating_gp_utag4_max', 'rating_gp_utag4_mean', 'rating_gp_utag4_median', 'rating_gp_utag4_min', 'rating_gp_utag5_count', 'rating_gp_utag5_nunique', 'rating_gp_utag5_max', 'rating_gp_utag5_mean', 'rating_gp_utag5_median', 'rating_gp_utag5_min', 'rating_gp_utag6_count', 'rating_gp_utag6_nunique', 'rating_gp_utag6_max', 'rating_gp_utag6_mean', 'rating_gp_utag6_median', 'rating_gp_utag6_min', 'rating_gp_utag7_count', 'rating_gp_utag7_nunique', 'rating_gp_utag7_max', 'rating_gp_utag7_mean', 'rating_gp_utag7_median', 'rating_gp_utag7_min', 'rating_gp_utag8_count', 'rating_gp_utag8_nunique', 'rating_gp_utag8_max', 'rating_gp_utag8_mean', 'rating_gp_utag8_median', 'rating_gp_utag8_min', 'graph_utmt_dislike_mid_gp_utid_count', 'graph_utmt_dislike_mid_gp_utid_nunique', 'graph_utmt_dislike_mid_gp_utid_diff', 'graph_utmt_dislike_mid_gp_utid_rate', 'graph_utmt_dislike_tid_freq', 'graph_utmt_like_mid_gp_utid_count', 'graph_utmt_like_mid_gp_utid_nunique', 'graph_utmt_like_mid_gp_utid_diff', 'graph_utmt_like_mid_gp_utid_rate', 'graph_utmt_like_tid_freq', 'graph_utut_count', 'graph_utut_nunique', 'graph_utut_nunique_rate', 'graph_utut_nunique_diff', 'graph_utut_freq']
params = {'objective':'binary',
          'boosting':'gbdt',          
          'metric':'auc',
          'learning_rate':0.1,
          'num_leaves': 3,          
          'min_data_in_leaf': 20,
          'lambda_l1':0,
          'lambda_l2':0,
          'num_threads': -1,
          'verbose':-1,
          'seed':666666}

train = dff[dff.set=='train']
tres,tgbm,valf_,score_mean=kflod_gbm(train,use_cols,params,k=8,y='y',esr=50)
print(np.mean(score_mean),np.mean(score_mean)-np.std(score_mean))

def get_val_score(valf2_,gbm_12,use_cols,pred='pred'):
    p_cols=[]
    for k,v in gbm_12.items():
        valf2_['p_{}'.format(k)] = v.predict(valf2_[use_cols])
        p_cols.append('p_{}'.format(k))
    valf2_[pred] = valf2_[p_cols].mean(axis=1)
    valf2_.drop(columns=p_cols,inplace=True)
    return valf2_

test = get_val_score(dff[dff.set=='test'],tgbm,use_cols)
test[['uid', 'tid', 'pred']].to_csv(data_path+'./sub/sub_8298.csv',index=False,sep=' ',header=False)