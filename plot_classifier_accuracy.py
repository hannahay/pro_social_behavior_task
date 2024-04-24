# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:04:12 2024

@author: HCattan
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import os  #to change directory
import scipy.io  #to load matlab files

os.chdir('D:\\Social decision making task\\processed_data\\t_test\\')  # change direction to the folder with results
os.getcwd()  # check the direction

mice=[ "ACC25", "ACC28", "ACC29", "ACC30","ACC23"]
days=[0,1,2,3,4]
param_svc=['linear', 'poly', 'rbf']
time_window_name=['baseline', 'before', 'onset', 'after']



for n,p in enumerate(time_window_name):
    
    fig1 =plt.figure(figsize=(40,35))

    for i,j in enumerate(param_svc):
        
    #filename_res=f'SVC_onset_{param_svc[sp]}.pkl'
        filename_res=f'SVC_{p}_equalnumoftrials_{j}.pkl'
    
        with open(filename_res, 'rb') as f:  # Python 3: open(..., 'rb')
            f_score, accuracy, confusion_matrix_all,perc_of_neurons,decision_pref_score= pickle.load(f)
    
    
        for M in range(len(mice)):
            ax1 = plt.subplot(3,2, M+1)
            plt.plot(accuracy[M,:])
            if M==0:
                plt.ylabel('accuracy',fontsize="25")
                plt.legend(param_svc, fontsize="25", loc='lower left')
            plt.xlabel('days', fontsize="25")
            plt.title(mice[M], fontsize="25")
            plt.rc('xtick', labelsize=25) 
            plt.rc('ytick', labelsize=25)
            plt.ylim(0,1)
    
    plt.show()        
    fig1 .savefig(f'SVC_{p}_equalnumoftrials_permouse.png')   
    plt.close()



for n,p in enumerate(time_window_name):
    
    fig2 =plt.figure(figsize=(40,35))
    
    for i,j in enumerate(param_svc):
        
    #filename_res=f'SVC_onset_{param_svc[sp]}.pkl'
        filename_res=f'SVC_{p}_equalnumoftrials_{j}.pkl'
    
        with open(filename_res, 'rb') as f:  # Python 3: open(..., 'rb')
            f_score, accuracy, confusion_matrix_all,perc_of_neurons,decision_pref_score= pickle.load(f)
    
        ax1 = plt.subplot(3,2, i+1)
    
        for M in range(len(mice)):
            plt.plot(accuracy[M,:])
            plt.ylabel('accuracy',fontsize="35")
            #plt.legend(param_svc, fontsize="40", loc='lower left')
            plt.xlabel('days', fontsize="40")
            plt.title(j, fontsize="40")
            plt.rc('xtick', labelsize=40) 
            plt.rc('ytick', labelsize=40) 
    
    plt.show()        
    fig2 .savefig(f'SVC_{p}_equalnumoftrials_per_svm.png')   
    plt.close()

