import pandas as pd
from sentence_transformers import SentenceTransformer
import gc
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from Training import run_experiment, run_experiment_XGB
from Evaluation import evaluate, evaluate_XGB
from utils import normalize, eng_class, sampling_k_elements, extract_graph
import numpy as np
import networkx as nx
from tensorflow import keras
from keras.utils import to_categorical
import random
from models.Xgboost import create_XGB
from models.Conv1D import create_Conv1D
from models.GAT import create_GAT
from models.GCN import create_GCN
from models.MLP import create_MLP
import argparse
import pickle
import os
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser("TweetGage Params")
    a = parser.add_argument
    a('--LOAD_CSV', action='store_true')
    a('--EXTRACT_BERT', action='store_true')
    a('--USE_PCA', action='store_true')
    a('--USER_FEAT', action='store_true')
    a('--BERT_FEAT', action='store_true')
    a('--Model_Type', default='GCN', type=str)
    return parser.parse_args()


def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(2)
    tf.random.set_seed(2)
    np.random.seed(2)
    random.seed(2)


def select_params(Model_type, X_train, y_train, X_test, y_test, df, g, num_classes=1, num_epochs=10):
    num_classes = num_classes
    num_epochs = num_epochs
    dropout_rate = None
    num_layers = None
    num_heads = None
    if Model_type == 'GCN':
        hidden_units = [16]
        dropout_rate = 0.3
        learning_rate = 0.1
        batch_size = 256
        input = np.array(X_train.index)
        target = y_train
        loss = keras.losses.MeanSquaredError() 
        optimizer = keras.optimizers.Adam
        input_test = np.array(X_test.index)
        target_test = y_test
        graph_info = extract_graph(g, df)
        model = create_GCN(graph_info, num_classes, hidden_units, dropout_rate)
    if Model_type == 'MLP':
        hidden_units = [32, 32]
        learning_rate = 0.01
        dropout_rate = 0.5
        batch_size = 256
        loss = keras.losses.MeanSquaredError() 
        input = X_train
        target = y_train
        input_test = X_test
        target_test = y_test
        optimizer = keras.optimizers.Adam
        model = create_MLP(X_train.shape[1], hidden_units, num_classes, dropout_rate)
    if Model_type == 'Conv1D':
        hidden_units = 64
        learning_rate = 0.1
        batch_size = 256
        model = create_Conv1D(num_classes, hidden_units, X_train.shape[1])
        input = X_train.values.reshape(-1, X_train.shape[1], 1)
        loss = keras.losses.MeanSquaredError() 
        target = y_train
        optimizer = keras.optimizers.Adam
        input_test = X_test
        target_test = y_test
    if Model_type == 'GAT':
        hidden_units = 100
        num_heads = 2
        num_layers = 1
        batch_size = 64
        learning_rate = 1e-2
        graph_info = extract_graph(g, df)
        input = X_train
        target = y_train
        model = create_GAT(graph_info[0], graph_info[1].T, hidden_units, num_heads, num_layers, num_classes)
        loss = keras.losses.MeanSquaredError() 
        optimizer = keras.optimizers.SGD
        input_test = X_test
        target_test = y_test
    if Model_type == 'XGBOOST':
        max_depth = 8
        learning_rate = 0.025
        subsample = 0.85
        colsample_bytree = 0.35
        eval_metric = 'rmse'
        objective = 'reg:squarederror'
        tree_method = 'gpu_hist'
        seed = 1
        model = create_XGB(max_depth, learning_rate, subsample,
                           colsample_bytree, eval_metric, objective,
                           tree_method, seed)
        return model
    return hidden_units, num_classes, learning_rate, num_epochs, dropout_rate, batch_size, num_layers, num_heads, input, target, loss, optimizer, input_test, target_test, model


def main(LOAD_CSV=False, EXTRACT_BERT=True, USE_PCA=False, USER_FEAT=True, BERT_FEAT=True, Model_Type='GCN'):
    reset_random_seeds()
    with open("network_tweets.pickle", "rb") as file:
        g= pickle.load(file)

    print("POST:", len(g.nodes))
    print("ARCS:", len(g.edges))
    print("COMPONENTS:", nx.number_connected_components(g))
    
    if not LOAD_CSV:
        df = pd.read_csv("./first_week.csv", lineterminator="\n")
        if EXTRACT_BERT:
            model = SentenceTransformer('efederici/sentence-bert-base')
            emb = model.encode(df["tweet"])
            
            df = pd.concat([df, pd.DataFrame(emb)], axis=1)
            del emb, model
            gc.collect()
            df.to_csv('first_week_posts_bert.csv', index=False)
    else:
        
        df = pd.read_csv("./first_week_posts_bert.csv")
        user_columns = ["no.of_hashtags", "no.of_mentions", "norm_likes_count", "norm_replies_count", "norm_retweet_count", "no.of_photos"]
        text_columns=[str(i) for i in range(768)]
        df_normalized = normalize(df[text_columns])

    selected_columns=user_columns+df_normalized.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(df[selected_columns], df["minmaxpop"], test_size=0.2,
                                                        random_state=42)


    if not Model_Type == 'XGBOOST':
        hidden_units, num_classes, learning_rate, num_epochs, dropout_rate, batch_size, num_layers, \
        num_heads, input, target, loss, optimizer, input_test, target_test, model = select_params(Model_Type, X_train,
                                                                                                  y_train, X_test,
                                                                                                  y_test,
                                                                                                  df,
                                                                                                  g,
                                                                                                  num_epochs=10)
        run_experiment(model, input, target, learning_rate, loss, num_epochs, batch_size, optimizer)
        evaluate(model, input_test, target_test)
    else:
        model = select_params(Model_Type, X_train, y_train, X_test, y_test, df, g,
                              num_epochs=10)
        obj = run_experiment_XGB(model, X_train, y_train)
        evaluate_XGB(obj, X_test, y_test)


if __name__ == '__main__':
    args = vars(parse_args())
    main(*list(args.values()))
