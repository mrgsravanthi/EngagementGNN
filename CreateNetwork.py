import pandas as pd
import ast
import networkx as nx
import datetime
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import numpy as np
import pickle
# settings (define the path to the tweets dataset)
delta = 15
path_to_tweets = "./first_week.csv"
n_jobs = 8

'''
Function to compute the edges on a slice of the tweets dataframe.
Returns the corresponding list of edges
'''

def edges_subset(split_df):
    edges = []
    for _, post in split_df.iterrows():
        sub_df = df.copy(deep=True)  # Remove the time-based filtering
        sub_df["connected"] = sub_df["hashtags"].apply(lambda x: len(set(x).intersection(post["hashtags"])))
        sub_df = sub_df.loc[sub_df["connected"] > 0]
        edges = edges + [(row["id"], post["id"], row["connected"]) for _, row in sub_df.iterrows()]
    return edges

# LOAD DATA
df = pd.read_csv(path_to_tweets, lineterminator='\n')
df = df.drop_duplicates('id')
df["hashtags"] = df["hashtags"].apply(lambda x: list(set(ast.literal_eval(x))))
df["time"] = pd.to_datetime(df["time"])

# COMPUTE EDGES using Parallel jobs. It works on dataframe splits
all_edges = Parallel(n_jobs=n_jobs)(delayed(edges_subset)(split_df) for split_df in tqdm(np.array_split(df, 100)))
all_edges = [y for x in all_edges for y in x]

# CREATE GRAPH
graph = nx.Graph()
# add weighted edges
graph.add_weighted_edges_from(all_edges)
# add isolated nodes
isolated = set(df["id"]).difference(list(graph.nodes))
graph.add_nodes_from(isolated)
# add node attributes
nx.set_node_attributes(graph, df.loc[df["id"].isin(list(graph.nodes))].set_index("id").to_dict(orient="index"))
print("NODES:", len(graph.nodes))
print("EDGES:", len(graph.edges))
print("DENSITY:", nx.density(graph))
print("NUM CONNECTED COMPONENTS:", len([len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)]))
print("MAX CONNECTED COMPONENT:", max([len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)]))

# check
if "/" in path_to_tweets:
    filename = path_to_tweets.split("/")[-1].split(".")[0]
else:
    filename = path_to_tweets.split("\\")[-1].split(".")[0]


with open("network_tweets.pickle", "wb") as file:
    pickle.dump(graph, file)
