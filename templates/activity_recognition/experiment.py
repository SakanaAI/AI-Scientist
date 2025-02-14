import argparse
import inspect
import json
import math
import pickle
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, \
                            ConfusionMatrixDisplay, \
                            accuracy_score, \
                            f1_score, \
                            recall_score, \
                            precision_score


def preprocess_data(target_type='hard'):
    # preprocess the data for activity recognition
    input_file_path = os.path.join('..', '..', 'data', 'activity_recognition', 'train')

    df_list = []
    for id in ['00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '00009', '00010']:
        # format the data (accelerometer and RSSI) for activity recognition
        df = pd.read_csv(os.path.join(input_file_path, id, 'acceleration.csv'))
        df.replace(np.nan, -120, inplace=True)
        df['t'] = pd.to_datetime(df['t'], unit='s')
        df.set_index('t', inplace=True)
        df = df.resample('1s').mean()
        
        # format the targe
        df_tgt = pd.read_csv(os.path.join(input_file_path, id, 'targets.csv'))
        df_tgt['t'] = pd.to_datetime(df_tgt['start'], unit='s')
        df_tgt.set_index('t', inplace=True)
        df_tgt.drop(['start', 'end'], axis=1, inplace=True)
        
        # add maximum target index column
        if target_type=='hard':
            for n, row in df_tgt.iterrows():
                df_tgt.loc[n, 'target'] = np.nan if np.any(np.isnan(row)) else np.argmax(row)

        # concatenate the data (accelerometer and RSSI) and the target
        df = pd.concat([df, df_tgt], axis=1, join='inner')
        df.dropna(inplace=True)

        # append to the df_list
        df_list.append(df)

    # concatenate the all data
    df = pd.concat(df_list, axis=0, ignore_index=True)

    # split feature dtata and labels
    data_id = ['x','y','z','Kitchen_AP', 'Lounge_AP', 'Upstairs_AP', 'Study_AP'] # if new features are added or removed, their column names should be added or removed here
    target_id = ['a_ascend', 'a_descend', 'a_jump', 'a_loadwalk' ,'a_walk',
                    'p_bent', 'p_kneel', 'p_lie', 'p_sit', 'p_squat', 'p_stand', 
                    't_bend', 't_kneel_stand', 't_lie_sit', 't_sit_lie', 't_sit_stand', 
                    't_stand_kneel', 't_stand_sit', 't_straighten','t_turn'] # if new labels are added or removed, their column names should be added or removed here
    if target_type=='soft':
        _id = target_id
    elif target_type=='hard':
        _id = ['target']
    else:
        raise ValueError('label_type should be either "hard" or "soft"')

    data = df[data_id].values
    target = df[_id].values
    
    return data, data_id, target, target_id


def get_classifier_grid():
    # Create cross-validation partitions from training
    # This should select the best set of parameters
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    clf = RandomForestClassifier()
    param_grid = {'n_estimators' : [200, 300, 500],
                  'min_samples_leaf': [5, 10, 20]}
    clf_grid = GridSearchCV(clf, 
                            param_grid=param_grid, 
                            cv=cv, 
                            refit=True,
                )
    return clf_grid

def split_train_test(X, y, partition=0):
    # Create train and test partitions
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        if i == partition:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
    return (X_train, y_train), (X_test, y_test)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_csv_file", 
        type=str, 
        default=os.path.join(
            os.path.dirname(__file__), 
            '../../data/activity_recognition/activity_recognition.csv'
        )
    )
    parser.add_argument("--out_dir", type=str, default="run_0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    config = parser.parse_args()

    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    # prepare the dataset
    dataset_name = 'SPHERE_Challenge' # at moment, only SPHERE_Challenge is supported
    data, features_id, target, target_id = preprocess_data(target_type='hard')
    (X_train, y_train), (X_test, y_test) = split_train_test(data, target)

    # get the classifier grid
    clf_grid = get_classifier_grid()
    
    # train the model
    start_time = time.time()
    clf_grid.fit(X_train, y_train)
    end_time = time.time()

    # predict the test data
    start_inf_time = time.time()
    y_pred = clf_grid.predict(X_test)
    end_inf_time = time.time()

    # post-process the prediction results
    y_pred_str = [target_id[int(i)] for i in y_pred]
    y_test_str = [target_id[int(i)] for i in y_test]

    # measure the performance
    accuracy = accuracy_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred, average='macro')
    # recall = recall_score(y_test, y_pred, average='macro')
    # precision = precision_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test_str, y_pred_str, labels=target_id)    
    
    # plot confusion matrix
    # ConfusionMatrixDisplay.from_predictions(
    #     y_test_str, y_pred_str, labels=target_id
    # )
    
    all_results = {}
    all_results[dataset_name] = {
        'accuracy': accuracy,
        # 'f1': f1,
        # 'recall': recall,
        # 'precision': precision,
        'confusion_matrix': cm.tolist(),
        'labels': target_id,
        'best_params': clf_grid.best_params_,
    }

    final_infos = {}
    final_infos[dataset_name] = {
            "means": {
                'training_time': end_time - start_time,
                'inferece_time': end_inf_time - start_inf_time,
                'accuracy': accuracy,
                'confusion_matrix': cm.tolist(),
            }
        }
    
    with open(os.path.join(config.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(os.path.join(config.out_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)