# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:49:11 2024

@author: HCattan
"""

import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import prepare_behavioral_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Example classifier, replace as needed
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier      
from sklearn.preprocessing import StandardScaler

# Specify the directory you want to add to the path
directory_to_add = r'D:\Social decision making task\Python code\python_scriptsandfunctions'
# Add the directory to the Python path
sys.path.append(directory_to_add)
from  prepare_behavioral_data_NEW import prepare_behavioral_data


# parameters to choose
equal_trials=1
recipient_features=0
experiment=1
previous_choice=0
num_of_shuffle=50

# variable to change according to your data
data_folder=r'D:\Social decision making task\processed_data\h5_files\actor_without_ms_exp101'
mice=['M113B', 'M114A', 'M114C', 'M115A', 'M115D', 'M118A','M120D', 'M125A', 'M125B', 'M127A', 'M127D', 'M128A']
empathic_port=['left', 'right', 'right', 'right', 'left', 'left', 'right', 'right', 'left', 'left','left','left' ]
altruistic_mice=[0,1,0,1,1,1,1,1]
days=[0,1,2,3,4]
time_window=np.linspace(-4,4,41)*1000 # from -4sec to 4sec around poking (in ms)
    


port_name=['empathic', 'selfish']
score_class=[]
score_class_shuffled=[]
score_days = pd.DataFrame()
score_days_shuffled = pd.DataFrame()


for day in range(len(days)-1) :
        
    score_class=[]
    score_class_shuffled=[]

    
    for t in range(0, len(time_window)-1, 2):
        df=pd.DataFrame()    
        for i in range (len(mice)-3):
            data_mouse=prepare_behavioral_data (data_folder, mice[i], day, empathic_port[i], time_window[t:t+2],
                                                equal_trials, recipient_features, previous_choice) 
            nan_values = ['NaN','nan', 'N/A', 'NULL', '']  # Add any other representations you need to handle
            # Remove rows with any NaN values and save it back to the same variable
            data_mouse.replace(nan_values, np.nan, inplace=True)
            data_mouse = data_mouse.dropna()            
            df=pd.concat([df, data_mouse])
        
        
        selfish_left = df[(df['labels'] == 'selfish') & (df['port_label'] == 'left')]
        selfish_right = df[(df['labels'] == 'selfish') & (df['port_label'] == 'right')]
        empathic_left = df[(df['labels'] == 'empathic') & (df['port_label'] == 'left')]
        empathic_right = df[(df['labels'] == 'empathic') & (df['port_label'] == 'right')]
        
        # Determine the size of the subsample (assuming equal numbers for each category)
        sample_size = min(len(selfish_left), len(selfish_right), len(empathic_left), len(empathic_right))
        
        # Sample randomly from each group
        random_df = pd.concat([
            selfish_left.sample(n=sample_size, replace=False),
            selfish_right.sample(n=sample_size, replace=False),
            empathic_left.sample(n=sample_size, replace=False),
            empathic_right.sample(n=sample_size, replace=False)
        ], axis=0)
            
            
        random_df.reset_index(drop=True, inplace=True)
        
        X=random_df.drop(["labels", "port_label"], axis=1)
        y =random_df["labels"]
        
        X_final, y_final = shuffle(X, y) 
        
        X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_train = pd.DataFrame(X_train, columns=X.columns)
        
        # Initialize and train a classifier (e.g., SVM)
        classifier = SVC(kernel='linear')
        classifier.fit(X_train, y_train)
        
        # Predict on the test set
        X_test = sc.transform(X_test)
        X_test = pd.DataFrame(X_test, columns=X.columns)

        y_pred = classifier.predict(X_test)    
        # Evaluate the classifier
        accuracy = accuracy_score(y_test, y_pred)    
        score_class.append(accuracy)

        
        del X_train 
        
        accuracy_shuffle=[]
        for m in range(num_of_shuffle):
            y_final_shuffled = y_final.sample(frac=1).reset_index(drop=True)
            X_train, X_test, y_train_shuffle, y_test_shuffle = train_test_split(X_final, y_final_shuffled, test_size=0.2)
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_train = pd.DataFrame(X_train, columns=X.columns)
            classifier.fit(X_train, y_train_shuffle)
            
            X_test = sc.transform(X_test)
            X_test = pd.DataFrame(X_test, columns=X.columns)
            y_pred = classifier.predict(X_test)    
            # Evaluate the classifier
            accuracy_shuffle.append( accuracy_score(y_test_shuffle, y_pred))
            
            
            
        score_class_shuffled.append(np.mean(accuracy_shuffle))
        
        
        del accuracy , accuracy_shuffle, X_train, y_train
        
    score_days[day]= score_class
    score_days_shuffled[day]= score_class_shuffled
    
    plt.figure    
    time=time_window[0:40:2]/1000
    plt.plot(time, score_class)    
    plt.plot(time,score_class_shuffled)  
    plt.title(f'day{day}')  
    plt.xlabel('time around poking (sec)')
    plt.ylabel('classifier f1 score')
    plt.show()
    
    



