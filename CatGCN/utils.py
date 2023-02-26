import torch

import time
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from texttable import Texttable

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    t.set_precision(6)
    print(t.draw())

def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    return graph

def field_reader(path):
    """
    Function to read the field index from the path.
    :param path: Path to the field index.
    :return field_index: Numpy matrix of field index.
    """
    field_index = np.load(path).astype(np.int64)
    return field_index

def target_reader(path):
    """
    Reading the target vector from disk.
    :param path: Path to the target.
    :return target: Target vector.
    """
    target = np.array(pd.read_csv(path).iloc[:,1]).reshape(-1,1)
    return target

def label_reader(path):
    """
    Reading the user_label file from the path.
    :param path: Path to the label file
    :return user_labels: User labels DataFrame file.
    """
    user_labels = pd.read_csv(path)
    return user_labels

def distr_label_attr(df, label, attr):
    """
    For a given df's label (e.g. gender), compute the distribution
    of a given attribute (e.g. age) for each label's class
    """
    return df.groupby([label, attr])[attr].count()

def pos_preds_attr_distr(df, targets, predictions, idx_list, label, attr):
    """
    Given a list of prediction, compute the given attribute's
    distribution for the correct predictions of the label
    """
    # Distribution of attribute's classes in test set
    df_test_grouped = df.iloc[idx_list].groupby([label, attr])[attr]
    dict_test_grouped = df_test_grouped.apply(list).to_dict()
    for k,v in dict_test_grouped.items():
        dict_test_grouped[k] = len(v)

    # Distribution of attribute's classes for correct predictions
    pos_preds = targets == predictions
    idx_pos_preds = idx_list[pos_preds]
    df_pos_preds = df.iloc[idx_pos_preds]
    df_pos_preds_grouped = df_pos_preds.groupby([label, attr])[attr]
    dict_pos_preds_grouped = df_pos_preds_grouped.apply(list).to_dict()
    for k,v in dict_pos_preds_grouped.items():
        dict_pos_preds_grouped[k] = len(v)

    # Compute correct prediction percentage
    dict_perc_preds = {}
    for k in dict_test_grouped.keys():
        try:
            perc = dict_pos_preds_grouped[k] / dict_test_grouped[k]
            dict_perc_preds[k] = perc
        except KeyError:
            dict_perc_preds[k] = 0

    return dict_perc_preds