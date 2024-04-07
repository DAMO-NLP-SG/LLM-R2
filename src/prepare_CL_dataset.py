# @TIME : 8/11/23 8:34 PM
# @AUTHOR : LZDH
import os
import math
import ast
import numpy as np
import random
import torch
import torch.nn as nn
import pandas as pd
from collections import deque
from sentence_transformers import SentenceTransformer
from get_query_meta import *
from encoder import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model = SentenceTransformer('all-MiniLM-L6-v2')


def conv_dict(in_dict):
    for key in list(in_dict.keys()):
        # print(in_dict[key])
        # print(type(in_dict[key]))
        if isinstance(in_dict[key], str):
            in_dict[key] = str(in_dict[key])
        else:
            in_dict[key] = str(in_dict[key].tolist())
    return str(in_dict)


def preprocess_dataset_train(dataset_df, model, dataset):
    dataset_0 = []
    db_ids = []
    for index, row in dataset_df.iterrows():
        db_ids.append(row['db_id'])
        dataset_0.append([row['query'], row['pos'], row['neg']])
    # print(dataset[0][0])
    dat_features = prepare_enc_data(dataset_0, model, db_ids)

    org = []
    pos = []
    neg = []
    for row in dat_features:
        org.append(conv_dict(row[0]))
        pos.append(conv_dict(row[1]))
        neg.append(conv_dict(row[2]))
    df_train = {'db_ids': db_ids,
              'org': org,
              'pos': pos,
              'neg': neg}
    df_train = pd.DataFrame(df_train)
    df_train.to_csv('../data/data_simcse/' + dataset + '/train_' + dataset + '_for_simcse.csv', header=False, index=False)


def preprocess_dataset_val(dataset_df_val, dataset_df_test, model, dataset):
    starter = '\t'.join(['main-captions', 'MSRvid', '2012test'])
    dataset_val = []
    db_ids_val = []
    labels_val = []
    for index, row in dataset_df_val.iterrows():
        db_ids_val.append(row['db_id'])
        dataset_val.append([row['query'], row['test']])
        labels_val.append(row['label'])
    dat_features_val = prepare_enc_data(dataset_val, model, db_ids_val)
    starter_ind = 10001
    vals = []
    for row_ind in range(len(dat_features_val)):
        vals.append('\t'.join(
            [starter, str(starter_ind)[-4:], str(labels_val[row_ind]), str(conv_dict(dat_features_val[row_ind][0])),
             str(conv_dict(dat_features_val[row_ind][1]))]))
        starter_ind += 1
    df_val = {'col': vals}
    df_val = pd.DataFrame(df_val)
    df_val.to_csv('../data/data_simcse/' + dataset + '/sts-dev.csv', header=False, index=False)

    dataset_test = []
    db_ids_test = []
    labels_test = []
    for index, row in dataset_df_test.iterrows():
        db_ids_test.append(row['db_id'])
        dataset_test.append([row['query'].replace(' value ', ' calcite_value '), row['test'].replace(' value ', ' calcite_value ')])
        labels_test.append(row['label'])
    print(dataset_test[0])
    dat_features_test = prepare_enc_data(dataset_test, model, db_ids_test)
    starter_ind = 10001
    tests = []
    for row_ind in range(len(dat_features_test)):
        tests.append('\t'.join(
            [starter, str(starter_ind)[-4:], str(labels_test[row_ind]), str(conv_dict(dat_features_test[row_ind][0])),
             str(conv_dict(dat_features_test[row_ind][1]))]))
        starter_ind += 1
    df_test = {'col': tests}
    df_test = pd.DataFrame(df_test)
    df_test.to_csv('../data/data_simcse/' + dataset + '/sts-test.csv', header=False, index=False)


dataset = 'dsb'
dataset_file = '../data/data_simcse/' + dataset + '/training_triplets_' + dataset + '_total.csv'
columns = ['db_id', 'query', 'pos', 'neg']
dataset_df = pd.read_csv(dataset_file, header=None, names=columns)
preprocess_dataset_train(dataset_df, model, dataset)

# test_columns = ['db_id', 'query', 'test', 'label']
# dataset_df_val = pd.read_csv('data_simcse/' + dataset + '/val_pairs_' + dataset + '.csv', header=None, names=test_columns)
# dataset_df_test = pd.read_csv('data_simcse/' + dataset + '/test_pairs_' + dataset + '.csv', header=None, names=test_columns)
# preprocess_dataset_val(dataset_df_val, dataset_df_test, model, dataset)
