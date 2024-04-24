# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:48:31 2024

@author: HCattan
"""

import pandas as pd
import numpy as np
import seaborn as sns
import os  #to change directory
import scipy.io  #to load matlab files
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split, cross_val_score , cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn import metrics

import pickle

os.chdir('D:\\Social decision making task\\processed_data\\t_test\\')  # change direction to the folder with results
os.getcwd()  # check the direction




param_svc=['linear']
time_window_name=['baseline', 'before', 'onset', 'after']
time_wind=np.array([[-2 ,-1], [-1,0], [0,1], [1,2]])

mice=["ACC23", "ACC25", "ACC28","ACC29", "ACC30"]
#mice=["ACC25_control","ACC28_control","ACC29_control", "ACC30_control"]

days=[0,1,2,3,4]
c=0
    
for t in range (time_wind.shape[0]):
    
    scores_all=np.empty(shape=[len(mice), len(days), len(param_svc)],dtype= 'float')
    scores_all[:]=np.nan
    score_all_shuffle=np.empty(shape=[len(mice), len(days), len(param_svc)],dtype= 'float')
    
    for l,p in enumerate (param_svc):


        confusion_matrix_all=np.zeros(shape=[len(mice), len(days)],dtype = 'float')
        f_score=np.zeros(shape=[len(mice), len(days)])
        accuracy=np.zeros(shape=[len(mice), len(days)])
        ind_before=np.zeros( shape=[len(mice), len(days),3],dtype = 'object' )
        ind_onset=np.zeros( shape=[len(mice), len(days),3],dtype = 'object' )
        ind_after=np.zeros( shape=[len(mice), len(days),3],dtype = 'object' )
        decision_pref_score=np.empty_like('array_like', shape=[len(mice),len(days)],dtype = 'float')
        num_of_neurons=np.empty_like('array_like', shape=[len(mice), len(days)],dtype = 'object')
        neurons_before=np.empty( shape=[len(mice), len(days),3],dtype = 'object' )
        neurons_onset=np.empty( shape=[len(mice), len(days),3],dtype = 'object' )
        neurons_after=np.empty( shape=[len(mice), len(days),3],dtype = 'object' )
        scores_control_i=np.empty( shape=10,dtype = 'float' )
        perc_of_neurons=np.empty_like('array_like', shape=[len(mice), len(days),2],dtype = 'float')

        for M in range (len(mice)):
            
            for D in range (len(days)):
                filename=f'{mice[M]}_day{days[D]}.mat'
                mat = scipy.io.loadmat(filename) 
                df = pd.DataFrame(mat['data_z'])
                df_pokes_t=pd.DataFrame(mat['pokes_ms'])
                df_neurons_avg_poke=pd.DataFrame(mat['neurons_avg'])
                FPS = int(mat['FPS'])
                start=time_wind[t,0]*FPS;
                stopp=time_wind[t,1]*FPS
        
                num_of_neurons[M,D]=mat['good_cells_ind'].astype('float').shape[0]
                index_all=(mat['ind_onset'], mat['ind_before'], mat['ind_sound']) 
                index_allp=np.concatenate(np.concatenate (index_all, axis=0), axis=0)  # concatenate
                ind_unique=set(np.squeeze(np.concatenate(index_allp, axis=1), axis=0))  # concatenate + reduce dimensions + unique number in list
                #neurons_of_interest=df[list(ind_unique),:]
                
                #ind_neuron_emp[M,D]=set(np.concatenate(mat['ind_onset'][0,0], mat['ind_before'][0,0], mat['ind_sound'][0,0]))
                #ind_neuron_ina[M,D]=set(np.concatenate(mat['ind_onset'][0,1], mat['ind_before'][0,1], mat['ind_sound'][0,1]))

                
                perc_of_neurons[M, D] = len(ind_unique) / num_of_neurons[M, D] * 100
                poke_ms_0_len = mat['pokes_ms'][0, 0].shape[1]
                poke_ms_1_len = mat['pokes_ms'][0, 1].shape[1]
                decision_pref_score[M, D] = (poke_ms_0_len - poke_ms_1_len) / (poke_ms_0_len + poke_ms_1_len) * 100
        
               
                for k in range(mat['ind_before'].shape[1]):
                    ind_before[M,D,k]=mat['ind_before'][0,k]-1;
                    neurons_before[M,D,k]=df_neurons_avg_poke.iloc[0,k][ind_before[M,D,k]][:,:,np.arange(start,2*start)]
                    ind_onset[M,D,k]=mat['ind_onset'][0,k]-1;
                    neurons_onset[M,D,k]=df_neurons_avg_poke.iloc[0,k][ind_onset[M,D,k]][:,:,np.arange(2*start,3*start)]
                    ind_after[M,D,k]=mat['ind_sound'][0,k]-1;
                    neurons_after[M,D,k]=df_neurons_avg_poke.iloc[0,k][ind_after[M,D,k]][:,:,np.arange(3*start,4*start)]
        
        
        
                emp=np.squeeze(mat['pokes_ms'][0,0], axis=0)  
                all_onset1=np.empty(shape=[df.shape[0],len(emp)])    
            
                for k in range(len(emp)):
                    all_onset1[:,k]=df.iloc[:,np.arange(emp[k]+start,emp[k]+stopp)].mean(axis=1)
         
         
                ina=np.squeeze(mat['pokes_ms'][0,1], axis=0)
                all_onset2=np.empty(shape=[df.shape[0],len(ina)])    
                 
                for k in range(len(ina)):
                    all_onset2[:,k]=df.iloc[:,np.arange(ina[k]+start,ina[k]+stopp)].mean(axis=1)
         
                equal_num_of_trials=min(all_onset1.shape[1], all_onset2.shape[1])
                
                if equal_num_of_trials<12:
                    f_score[M,D]=np.nan
                    accuracy[M,D]=np.nan
                    c+=1
                    print(f'{c}')
                    
                else:
                    
                        
                    ind_onset1=pd.Series(random.choices(np.arange(all_onset1.shape[1]),k=equal_num_of_trials))
                    ind_onset2=random.choices(np.arange(all_onset2.shape[1]),k=equal_num_of_trials)                
                    all_onset=np.concatenate([all_onset1[:,ind_onset1], all_onset2[:,ind_onset2]], axis=1) 
                  
                    true_score = np.concatenate([np.ones(equal_num_of_trials), np.zeros(equal_num_of_trials)]).T

                    all_onset= all_onset.transpose()
               
            # find indexes with Nan and remove them         
                    ind_Nan=np.argwhere(np.isnan(all_onset))
                    if len(ind_Nan)>1:
                        all_onset= np.delete(all_onset,ind_Nan,axis=1 )
                
                    
                # try PCA and classfier for TIME WINDOW [0,1]sec (onset) for one mouse, one day
                
            
                    X_train=all_onset
                    y_train=true_score
            
                    sc = StandardScaler()
                    X_train = sc.fit_transform(X_train)
            
                    pca = PCA(n_components = min(X_train.shape))
                    X_train = pca.fit_transform(X_train)
            
                    explained_variance = pca.explained_variance_ratio_
                    num_of_relevantPCA=np.argwhere(np.cumsum(explained_variance)<0.85)
                    X_trainA=np.squeeze(X_train[:,num_of_relevantPCA])
            
           
                    clf = svm.SVC(kernel=p)
                    
                    scores = cross_val_score(clf, X_trainA, y_train, cv=10)
                    scores_all[M,D]=np.mean(scores)
                    
                    for i in range (10):
                        true_score_control=np.array(random.choices(true_score,k=true_score.shape[0] ))
                        scores_control_i[i] = np.mean(cross_val_score(clf, X_trainA, true_score_control, cv=10))
                    
                    score_all_shuffle[M,D,l]=np.mean(scores_control_i)
                    
            
                    del pca, X_train, y_train, X_trainA, num_of_relevantPCA, clf,explained_variance, all_onset, equal_num_of_trials, all_onset2, all_onset1
            
            
                
            with open(f'scores_{time_window_name[t]}','wb') as f: 
                    pickle.dump([scores_all, score_all_shuffle], f)
    
    
               