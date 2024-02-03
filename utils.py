import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import networkx as nx
import pandas as pd

def eng_class(x):
    if x <= 0:
        return 0
    else:
        return 1


def sampling_k_elements(group, k=30000):
    if len(group) < k:
        return group
    return group.sample(k)


def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)



from sklearn.preprocessing import MinMaxScaler

def normalize(df):
    # Create a MinMaxScaler instance
    scaler = MinMaxScaler()
    for col in df.columns:
        # Attempt to convert the column to numeric values
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            # Handle the case where conversion to numeric fails (e.g., non-numeric values)
            print(f"Warning: Could not convert column '{col}' to numeric values.")
            continue
        
    # Iterate through columns and normalize numeric columns
    for col in df.columns:
        # Check if the column contains numeric values
        if pd.api.types.is_numeric_dtype(df[col]):
            # Reshape the column to a 2D array before scaling
            column_data = df[col].values.reshape(-1, 1)

            # Fit and transform the data using MinMaxScaler
            scaled_data = scaler.fit_transform(column_data)

            # Assign the scaled values back to the DataFrame
            df.loc[:, col] = scaled_data.flatten()

    return df

def extract_graph(g, df):
    mapping_graph = {k: v for v, k in enumerate(g.nodes)}
    g = nx.relabel_nodes(g, mapping_graph)
    edges = np.array(list(g.edges)).T
    edges_weight = [x[2]["weight"] for x in g.edges(data=True)]
    user_columns = ["no.of_hashtags", "no.of_mentions", "norm_likes_count", "norm_replies_count", "norm_retweet_count", "no.of_photos"]
    text_columns=[str(i) for i in range(768)]
    features_names=user_columns+text_columns
    node_features = tf.cast(
        df.sort_index()[features_names].to_numpy(), dtype=tf.dtypes.float32
    )

    graph_info = (node_features, edges, edges_weight)
    return graph_info
