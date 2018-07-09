# -*- coding:utf-8 -*-
from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import surprise
from surprise import KNNBaseline, Reader
from surprise import Dataset
from surprise import evaluate, print_perf
import csv
from surprise import SVD,SVDpp
from surprise import  GridSearch
from surprise import NMF
from pandas import Series
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    csv_reader = csv.reader(open('neteasy_playlist_id_to_name_data.csv',encoding='utf-8'))
    id_name_dict = {}
    name_id_dict = {}
    for row in csv_reader:
        id_name_dict[row[0]] = row[1]
        name_id_dict[row[1]] = row[0]
    csv_reader = csv.reader(open('neteasy_song_id_to_name_data.csv',encoding='utf-8'))
    song_id_name_dict = {}
    song_name_id_dict = {}
    for row in csv_reader:
        song_id_name_dict[row[0]] = row[1]
        song_name_id_dict[row[1]] = row[0]
    
    file_path = os.path.expanduser('neteasy_playlist_recommend_data.csv')
    reader = Reader(line_format='user item rating timestamp', sep=',')
    music_data = Dataset.load_from_file(file_path, reader=reader)
    music_data.split(n_folds=5)
    print('构建数据集')
    trainset = music_data.build_full_trainset()
    
    
    param_grid = { 'n_factors':range(10,30,2), 'n_epochs': [10,15,20], 'lr_all': [0.002, 0.005, 0.1],'reg_all': [0.4, 0.6, 0.8]}
    param_grid = { 'n_factors':range(2,22,2), 'n_epochs': [10], 'lr_all': [0.1],'reg_all': [0.4]}
    param_grid = { 'n_factors':[2], 'n_epochs':range(11), 'lr_all': [0.1],'reg_all': [0.4]}
    grid_search = GridSearch(SVDpp, param_grid, measures=['RMSE', 'MAE'])
    grid_search.evaluate(music_data)    
    print(grid_search.best_params['RMSE'])   
    print(grid_search.best_params['MAE'])
   
    # 开始训练模型
    print('开始训练模型...')
    #algo = KNNBaseline()
    algo = SVDpp(n_factors=grid_search.best_params['RMSE']['n_factors'],n_epochs=grid_search.best_params['RMSE']['n_epochs'],lr_all=grid_search.best_params['RMSE']['lr_all'],reg_all=grid_search.best_params['RMSE']['reg_all'],verbose=2)
    algo=SVDpp()
    #algo=SVD()
    #algo=SVDpp()
    perf = evaluate(algo, music_data, measures=['RMSE', 'MAE'],verbose=1)
    
    print_perf(perf)
    
    #print()
    #print('针对歌单进行预测:')
    #current_playlist_name =list(name_id_dict.keys())[3]
    #print('歌单名称', current_playlist_name)

    #playlist_rid = name_id_dict[current_playlist_name]
    #print('歌单rid', playlist_rid)

    #playlist_inner_id = algo.trainset.to_inner_uid(playlist_rid)
    #print('歌曲inid', playlist_inner_id)

    #algo.compute_similarities()
    #playlist_neighbors_inner_ids = algo.get_neighbors(playlist_inner_id, k=10)
    #playlist_neighbors_rids = (algo.trainset.to_raw_uid(inner_id) for inner_id in playlist_neighbors_inner_ids)
    #playlist_neighbors_names = (id_name_dict[rid] for rid in playlist_neighbors_rids)

    #print()
    #print('歌单 《', current_playlist_name, '》 最接近的10个歌单为: \n')
    #for playlist_name in playlist_neighbors_names:
    #    print(playlist_name, algo.trainset.to_inner_uid(name_id_dict[playlist_name]))

    print()
    print('针对用户进行预测:')
    user_inner_id = 300
    print('用户内部id', user_inner_id)
    user_rating = trainset.ur[user_inner_id]
    print('用户评价过的歌曲数量', len(user_rating))
    items = map(lambda x:x[0], user_rating)
    real_song_id=[]
    real_song_name=[]
    for song in items:
        real_song_id.append(algo.trainset.to_raw_iid(song))
        real_song_name.append(song_id_name_dict[algo.trainset.to_raw_iid(song)])
        
    t_l=10
    song_list1=list(song_id_name_dict.keys())
    rank=[]
    for song in song_list1:
        rank.append(algo.predict(str(user_inner_id), str(song))[3])
    rank=Series(rank)
    rank1=rank.sort_values(ascending=False)
    predict_song_id=[]
    predict_song_name=[]
    for i in range(t_l):
        predict_song_id.append(song_list1[list(rank1.index)[i]])
        predict_song_name.append(song_id_name_dict[song_list1[list(rank1.index)[i]]])
#from pandas import Series
    a=Series(real_song_name)
    b=Series(predict_song_name)
    c=pd.DataFrame({'real':a,'predict':b})
    
    #t_l=20   #取top的长度
    #if len(user_rating)<=t_l:
    #    pre_song=list(rank1.index[range(t_l)])
    #    real_song=
    #    correct=
  
surprise.dump.dump('./knn_baseline.model', algo=algo)
MAE=[]
RMSE=[]
for i in range(10):
    MAE.append( grid_search.cv_results['scores'][i]['MAE'])
    RMSE.append( grid_search.cv_results['scores'][i]['RMSE'])
x=range(2,22,2)
plt.plot(x,MAE,label='MAE')
plt.plot(x,RMSE,label='RMSE')
plt.legend() 
plt.axis([0,22,0.5,1])  


x=range(2,22,2)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x,MAE)
ax1.set_ylabel('MAE')
ax1.set_title("MAE & RMSE")
ax1.set_xticks(x)
ax1.set_xlabel('n_factors')
ax1.legend()
ax2 = ax1.twinx()  # this is the important function
ax2.plot(x, RMSE, 'r')
ax2.set_ylabel('RMSE')

plt.show()


MAE=[]
RMSE=[]
for i in range(11):
    MAE.append( grid_search.cv_results['scores'][i]['MAE'])
    RMSE.append( grid_search.cv_results['scores'][i]['RMSE'])
x=range(1,12,1)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x,MAE)
ax1.set_ylabel('MAE')
ax1.set_title("MAE & RMSE")
ax1.set_xticks(x)
ax1.set_xlabel('epoch')
ax1.legend()
ax2 = ax1.twinx()  # this is the important function
ax2.plot(x, RMSE, 'r')
ax2.set_ylabel('RMSE')

plt.show()
