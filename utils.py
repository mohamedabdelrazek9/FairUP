## Todo utils for data pre-process

import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import re
from alibaba_processing.ali_CatGCN_pre_processing import get_count, filter_triplets, col_map, label_map

def load_networkx_file(model_type, data_extension, dataset_name, dataset_path, dataset_user_id_name, onehot_bin_columns, onehot_cat_columns, sens_attr, predict_attr):

    # load data from graphml to csv
    #print('Loading dataset for FairGNN...')

    #print(data_extension)
    print('Extracting networkx data format...')
    if data_extension == '.graphml':
        data = nx.read_graphml(dataset_path)
    elif data_extension == '.gexf':
        data = nx.read_gexf(dataset_path)
    elif data_extension == '.gml':
        data = nx.read_gml(dataset_path)
    elif data_extension == '.leda':
        data = nx.read_leda(dataset_path)
    elif data_extension == '.net':
        data = nx.read_pajek(dataset_path)
        
    # load graph nodes
    #print('Data extension', data_extension)
    #print('Data', data)
    df_nodes = pd.DataFrame.from_dict(dict(data.nodes(data=True)), orient='index')
    
    # check if user_id column is not assigned as the index
    if df_nodes.columns[0] != dataset_user_id_name:    
        # if so, then we make it as the first column
        df_nodes = df_nodes.reset_index(level=0)
        df_nodes = df_nodes.rename(columns={"index": dataset_user_id_name})

    # check if user_id column is not string
    if type(df_nodes[dataset_user_id_name][0]) != np.int64:
        # if so, we convert it to int
        df_nodes[dataset_user_id_name] = pd.to_numeric(df_nodes[dataset_user_id_name])
        df_nodes = df_nodes.astype({dataset_user_id_name: int})

    #check if sens_attr and predict_attr is float 
    if (df_nodes[sens_attr].dtype == np.float64):
        df_nodes[sens_attr] = df_nodes[sens_attr].astype(int)
    if (df_nodes[predict_attr].dtype == np.float64):
        df_nodes[predict_attr] = df_nodes[predict_attr].astype(int)


    # todo if dataset will be used for RHGN or CatGCN then return, else we assume for FairGNN then complete the onehot encoding process
    if model_type == 'RHGN':
        return df_nodes

    elif model_type == 'CatGCN':
        if dataset_name == 'nba' or dataset_name == 'pokec_z' or dataset_name == 'pokec_n':
            df_edge_list = nx.to_pandas_edgelist(data)
            return df_nodes, df_edge_list
        else: ## Data is either Alibaba or tecent and they will get their edges later on
            df_edge_list = None
            return df_nodes, df_edge_list

    else: # FairGNN
        if dataset_name == 'alibaba' or dataset_name == 'tecent':
            if dataset_name == 'tecent':
                df_nodes = bin_age_range_tecent(df_nodes)
                df_nodes = df_nodes.drop(columns=["cid1_name", "cid2_name ", "cid3_name", "item_name", "seg_name"])
            if dataset_name == 'alibaba':
                df_nodes = bin_alibaba(df_nodes)
            edges_path = create_edges(df_nodes, dataset_name)
            df_edge_list = edges_path
        # todo add one-hot encoding
        # add binary onehot encoding if needed
        if onehot_bin_columns is not None:
            df_nodes = apply_bin_columns(df_nodes, onehot_bin_columns)
        # add categorical onehot encoding if needed
        if onehot_cat_columns is not None:
            df_nodes = apply_cat_columns(df_nodes, onehot_cat_columns)

        if dataset_name == 'nba' or dataset_name == 'pokec':
            # load graph edges
            df_edge_list = nx.to_pandas_edgelist(data)

        #save the edges as .txt file
        edges_path = './FairGNN_data_relationship'
        df_edge_list.to_csv(r'{}.txt'.format(edges_path), header=None, index=None, sep=' ', mode='a')

        return df_nodes, edges_path


def load_neo4j_file(model_type, dataset_path, dataset_name, uneeded_columns, onehot_bin_columns, onehot_cat_columns):
    # todo pre-process node and edge data
    #print('Loading dataset for FairGNN...')
    print('Extracting neo4j data format...')
    df = pd.read_json(dataset_path, lines=True) # may cause error

    # todo extract node csv
    nodes_df = df.loc(df['type'] == ['node'])
    #delete un-needed column
    nodes_df = nodes_df.drop(['label', 'start', 'end'], axis=1)

    # get nodes properties as list of json
    prop_list = []
    id_list = []
    labels_list = []
    for index, row in nodes_df.iterrows():
        prop_list.append(row['propertiees'])
        id_list.append(row['id'])
        labels_list.append(row['labels'])

    for i in range(len(prop_list)):
        prop_list[i]['id'] = id_list[i]
        prop_list[i]['labels'] = labels_list[i]

    # create new csv from the prop list
    new_nodes_df = pd.DataFrame(prop_list)
    new_nodes_df = new_nodes_df.drop(['properties'], axis=1)


    # make id as first column
    first_column = new_nodes_df.pop('id')
    new_nodes_df.insert(0, 'id', first_column)

    # we only apply the uneeded columns feature and the onehot encoding for the the FairGNN model 
    if model_type == 'FairGNN':
        # add binary onehot encoding if needed
        if onehot_bin_columns is not None:
            new_nodes_df = apply_bin_columns(new_nodes_df, onehot_bin_columns)
        # add categorical onehot encoding if needed
        if onehot_cat_columns is not None:
            new_nodes_df = apply_cat_columns(new_nodes_df, onehot_cat_columns)

        # todo remove columns that we don't want to have in the dataframe
        if len(uneeded_columns) == 0:
            new_nodes_df = remove_column_from_df('description') ## we don't want descriptions in our code per default
        else:
            new_nodes_df = remove_column_from_df(uneeded_columns) ## user defined columns 

        # now we remove columns that we don't want it to change for the next step (one-hot step) (e.g. id, person id)
        new_nodes_df = remove_unneeded_columns(new_nodes_df)
        
        # replace nan with 0
        new_nodes_df = new_nodes_df.replace(r'^\s*$', np.nan, regex=True)
        new_nodes_df = new_nodes_df.fillna(0)

    # Todo know which columns to filter out 
    # not needed -- replacment the function apply_cat_columns
    #new_nodes_df = apply_one_hot_encodding(nodes_columns, new_nodes_df)

############################################
    #extract edges relationships
    if dataset_name == 'alibaba' or dataset_name == 'tecent':
        return new_nodes_df
    else:
        edges_df = df.loc[(df['type'] == 'relationship')]
        edges_df = edges_df.drop(['labels'], axis=1)

        edges_relation = pd.DataFrame(columns=['start', 'end'], index=range(len(edges_df.index)))
        i = 0

        for index, row in edges_df.iterrows():
            edges_relation['start'][i] = row['start']['id']
            edges_relation['end'][i] = row['end']['id']
            i = i+1 

        edges_relation.columns = [''] * len(edges_relation.columns)

    # save .txt
    # todo maybe return it normally?
    edges_path = './FairGNN_data_relationship'
    edges_relation.to_csv(r'{}.txt'.format(edges_path), sep='\t', header=False, index=False)

    return new_nodes_df, edges_relation


def remove_column_from_df(column, df):
    nodes_columns = df.columns.tolist()
    # check if we have list of columns or not
    if type(column) == list:
        for i in column:
            df = df.drop([i], axis=1)
    else:
        for c in nodes_columns:
            if c == column:
                df = df.drop([column], axis=1)


def remove_unneeded_columns(new_nodes_df):
    unneeded_columns = []
    nodes_columns = new_nodes_df.columns.tolist()

    matchers = ['id', 'iD', 'Id', 'name']
    matching = [s for s in nodes_columns if any(xs in s for xs in matchers)]

    for i in range(len(matching)):
        if matching[i].endswith('id') or matching[i].endswith('Id'):
            unneeded_columns.append(matching[i])
            nodes_columns.remove(matching|[i])

        if matching[i] == 'name':
            nodes_columns.remvoe(matching[i])

    nodes_columns.remove('id')
    nodes_columns.remove('labels')

    return nodes_columns


def apply_one_hot_encodding(nodes_columns, new_nodes_df):

    for column in nodes_columns:
        if new_nodes_df[column].dtype != 'int64' or new_nodes_df[column].dtype != 'float64':
            new_nodes_df[column] = new_nodes_df[column].apply(lambda x: ",".join(x) if isinstance(x, list) else x)

        tempdf = pd.get_dummies(new_nodes_df[column], prefix=column, drop_first=True)
        new_nodes_df = pd.merge(left=new_nodes_df, right=tempdf, left_index=True, right_index=True)

        new_nodes_df = new_nodes_df.drop(columns=column)

    new_nodes_df.columns = new_nodes_df.columns.str.replace(' \t', '')
    new_nodes_df.columns = new_nodes_df.columns.str.strip().str.replace(' ', '_')
    new_nodes_df.columns = new_nodes_df.columns.str.replace('___', '_')
    new_nodes_df.columns = new_nodes_df.columns.str.replace('__', '_')


    return new_nodes_df


def fair_metric(output,idx, labels, sens):
    #output == target
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()]==0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()]==1

    
    # parameters for "overall accuracy equality"
    #true_y = np.asarray(output)
    #true_y = output.detach().numpy()
    #true_y = np.asarray(true_y)
    #  Use tensor.detach().numpy()
    
    #y0_s0 = np.bitwise_and(true_y == 0, idx_s0)
    #y0_s1 = np.bitwise_and(true_y == 0, idx_s1)
    #y1_s0 = np.bitwise_and(true_y == 1, idx_s0)
    #y1_s1 = np.bitwise_and(true_y == 1, idx_s1)
    

    idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1)
    idx_s0_y0 = np.bitwise_and(idx_s0,val_y==0)
    idx_s1_y0 = np.bitwise_and(idx_s1,val_y==0)

    pred_y = (output[idx].squeeze()>0).type_as(labels).cpu().numpy()
    #parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    parity = np.abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    print('parity debug')
    print('pred_y:',pred_y)
    print('pred_y[idx_s0]:', pred_y[idx_s0])
    print('idx_s0:', idx_s0)
    print('parity:', parity)
    #equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))
    equality = np.abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))
    
    # treatment equality
    te1_s0 = (sum(pred_y[idx_s0_y0]) / sum(idx_s0_y0)) / (np.count_nonzero(pred_y[idx_s0_y1] == 0) / sum(idx_s0_y1))
    te1_s1 = (sum(pred_y[idx_s1_y0]) / sum(idx_s1_y0)) / (np.count_nonzero(pred_y[idx_s1_y1] == 0) / sum(idx_s1_y1))
    te_diff_1 = te1_s0 - te1_s1
    abs_ted_1 = abs(te_diff_1)

    te0_s0 = (np.count_nonzero(pred_y[idx_s0_y1] == 0) / sum(idx_s0_y1)) / (sum(pred_y[idx_s0_y0]) / sum(idx_s0_y0))
    te0_s1 = (np.count_nonzero(pred_y[idx_s1_y1] == 0) / sum(idx_s1_y1)) / (sum(pred_y[idx_s1_y0]) / sum(idx_s1_y0))
    te_diff_0 = te0_s0 - te0_s1
    abs_ted_0 = abs(te_diff_0)

    if abs_ted_0 < abs_ted_1:
        te_s0 = te0_s0
        te_s1 = te0_s1
        te_diff = te_diff_0
    else:
        te_s0 = te1_s0
        te_s1 = te1_s1
        te_diff = te_diff_1
    
    # "overall accuracy equality"
    oae_s0 = np.count_nonzero(pred_y[idx_s0_y0] == 0) / sum(idx_s0_y0) + sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1)
    oae_s1 = np.count_nonzero(pred_y[idx_s1_y0] == 0) / sum(idx_s1_y0) + sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1)
    oae_diff = np.abs(oae_s0 - oae_s1) 

    # disparate_impact


    return parity, equality,oae_diff, te_diff


def apply_bin_columns(df, onehot_bin_columns):
    for column in df:
        if column in onehot_bin_columns:
            df[column] = df[column].astype(int)

    return df

def apply_cat_columns(df, onehot_cat_columns):
    df = pd.get_dummies(df, columns=onehot_cat_columns)

    return df

def create_edges(df_nodes, dataset_name):

    if dataset_name == 'alibaba':
        # divide data
        df_user = df_nodes[['userid', 'final_gender_code', 'age_level', 'pvalue_level', 'occupation', 'new_user_class_level ']].copy()
        df_item = df_nodes[['adgroup_id', 'cate_id']].copy()
        df_click = df_nodes[['userid', 'adgroup_id', 'clk']].copy()

        df_user.dropna(inplace=True)
        df_user.rename(columns={'userid':'uid', 'final_gender_code':'gender','age_level':'age', 'pvalue_level':'buy', 'occupation':'student', 'new_user_class_level ':'city'}, inplace=True)

        df_item.rename(columns={'adgroup_id':'pid','cate_id':'cid'}, inplace=True)

        df_click.rename(columns={'userid':'uid','adgroup_id':'pid'}, inplace=True)
        df_click = df_click[df_click['clk']>0]
        df_click.drop('clk', axis=1, inplace=True)
        df_click = df_click[df_click['uid'].isin(df_user['uid'])]
        df_click = df_click[df_click['pid'].isin(df_click['pid'])]

        df_click.drop_duplicates(inplace=True)

        uid_pid, uid_activity, pid_popularity = filter_triplets(df_click, 'uid', 'pid', min_uc=0, min_sc=2) # min_sc>=2
        #sparsity = 1. * uid_pid.shape[0] / (uid_activity.shape[0] * pid_popularity.shape[0])

        uid_pid_cid = pd.merge(uid_pid, df_item, how='inner', on='pid')
        raw_uid_cid = uid_pid_cid.drop('pid', axis=1, inplace=False)
        raw_uid_cid.drop_duplicates(inplace=True)

        uid_cid, uid_activity, cid_popularity = filter_triplets(raw_uid_cid, 'uid', 'cid', min_uc=0, min_sc=2) # min_sc>=2
        #sparsity = 1. * uid_cid.shape[0] / (uid_activity.shape[0] * cid_popularity.shape[0])

        uid_pid = uid_pid[uid_pid['uid'].isin(uid_cid['uid'])]
        uid_pid_1 = uid_pid[['uid','pid']].copy()
        uid_pid_1.rename(columns={'uid':'uid1'}, inplace=True)
        uid_pid_2 = uid_pid[['uid','pid']].copy()
        uid_pid_2.rename(columns={'uid':'uid2'}, inplace=True)

        uid_pid_uid = pd.merge(uid_pid_1, uid_pid_2, how='inner', on='pid')
        uid_uid = uid_pid_uid.drop('pid', axis=1, inplace=False)
        uid_uid.drop_duplicates(inplace=True)

        del uid_pid_1, uid_pid_2, uid_pid_uid

        # map
        user_label = df_user[df_user['uid'].isin(uid_cid['uid'])]
        uid2id = {num: i for i, num in enumerate(user_label['uid'])}
        cid2id = {num: i for i, num in enumerate(pd.unique(uid_cid['cid']))}

        user_label = col_map(user_label, 'uid', uid2id)
        user_label = label_map(user_label, user_label.columns[1:])

        user_edge = uid_uid[uid_uid['uid1'].isin(uid_cid['uid'])]
        user_edge = user_edge[user_edge['uid2'].isin(uid_cid['uid'])]

        user_edge = col_map(user_edge, 'uid1', uid2id)
        user_edge = col_map(user_edge, 'uid2', uid2id)

        return user_edge

    elif dataset_name == 'tecent':
        df_user = df_nodes[['user_id', 'gender', 'age_range']].copy()
        df_user.dropna(inplace=True)
        df_user.rename(columns={"user_id":"uid", "age_range":"age"}, inplace=True)

        df_item = df_nodes[['item_id', 'cid3']].copy()
        df_item.dropna(inplace=True)
        df_item.rename(columns={"item_id":"pid", "cid3":"cid"}, inplace=True)
        df_item.reset_index(drop=True, inplace=True)

        df_click = df_nodes[['user_id', 'item_id']].copy()
        df_click.dropna(inplace=True)
        df_click.rename(columns={"user_id":"uid", "item_id":"pid"}, inplace=True)
        df_click.reset_index(drop=True, inplace=True)

        df_item = df_item.sample(frac=0.15, random_state=11)
        df_item.reset_index(drop=True, inplace=True)

        df_click = df_click.sample(frac=0.15, random_state=11)
        df_click.reset_index(drop=True, inplace=True)

        df_click = df_click[df_click["uid"].isin(df_user["uid"])]
        df_click = df_click[df_click["pid"].isin(df_item["pid"])]

        df_click.drop_duplicates(inplace=True)
        df_click.reset_index(drop=True, inplace=True)

        df_click, uid_activity, pid_popularity = filter_triplets(df_click, 'uid', 'pid', min_uc=0, min_sc=2)
        sparsity = 1. * df_click.shape[0] / (uid_activity.shape[0] * pid_popularity.shape[0])

        df_click_item = pd.merge(df_click, df_item, how="inner", on="pid")
        raw_click_item = df_click_item.drop("pid", axis=1, inplace=False)
        raw_click_item.drop_duplicates(inplace=True)

        df_click_item, uid_activity, cid_popularity = filter_triplets(raw_click_item, 'uid', 'cid', min_uc=0, min_sc=2)
        sparsity = 1. * df_click_item.shape[0] / (uid_activity.shape[0] * cid_popularity.shape[0])

        df_click = df_click[df_click["uid"].isin(df_click_item["uid"])]
        df_click_1 = df_click[["uid", "pid"]].copy()
        df_click_1.rename(columns={"uid":"uid1"}, inplace=True)
        df_click_2 = df_click[["uid", "pid"]].copy()
        df_click_2.rename(columns={"uid":"uid2"}, inplace=True)

        df_click1_click2 = pd.merge(df_click_1, df_click_2, how="inner", on="pid")
        df_uid_uid = df_click1_click2.drop("pid", axis=1, inplace=False)
        df_uid_uid.drop_duplicates(inplace=True)

        del df_click_1, df_click_2, df_click1_click2

        # map
        df_label = df_user[df_user["uid"].isin(df_click_item["uid"])]
        uid2id = {num: i for i, num in enumerate(df_label['uid'])}
        cid2id = {num: i for i, num in enumerate(pd.unique(df_click_item['cid']))}

        df_label = col_map(df_label, 'uid', uid2id)
        df_label = label_map(df_label, df_label.columns[1:])

        user_edge = df_uid_uid[df_uid_uid['uid1'].isin(df_click_item['uid'])]
        user_edge = user_edge[user_edge['uid2'].isin(df_click_item['uid'])]

        user_edge = col_map(user_edge, 'uid1', uid2id)
        user_edge = col_map(user_edge, 'uid2', uid2id)

        return user_edge


def bin_age_range_tecent(df_nodes):
    age_dic = {'11~15':0, '16~20':0, '21~25':0, '26~30':1, '31~35':1, '36~40':2, '41~45':2, '46~50':3, '51~55':3, '56~60':4, '61~65':4, '66~70':4, '71~':4}
    df_nodes[["age_range"]] = df_nodes[["age_range"]].applymap(lambda x:age_dic[x])
    #df_nodes.rename(columns={"user_id":"uid", "age_range":"age"}, inplace=True)

    #df_nodes["bin_age"] = df_nodes["age"]
    df_nodes["age_range"] = df_nodes["age_range"].replace(1,0)
    df_nodes["age_range"] = df_nodes["age_range"].replace(2,1)
    df_nodes["age_range"] = df_nodes["age_range"].replace(3,1)
    df_nodes["age_range"] = df_nodes["age_range"].replace(4,1)


    return df_nodes

def bin_alibaba(df_nodes):
    df_nodes["age_level"] = df_nodes["age_level"].replace(1,0)
    df_nodes["age_level"] = df_nodes["age_level"].replace(2,0)
    df_nodes["age_level"] = df_nodes["age_level"].replace(3,0)
    df_nodes["age_level"] = df_nodes["age_level"].replace(4,1)
    df_nodes["age_level"] = df_nodes["age_level"].replace(5,1)
    df_nodes["age_level"] = df_nodes["age_level"].replace(6,1)


    df_nodes['pvalue_level'] = df_nodes['pvalue_level'].replace(3.0, 2.0)
    df_nodes['pvalue_level'] = df_nodes['pvalue_level'].astype('int64')

    return df_nodes


def calculate_dataset_fairness(df, dataset_name, sens_attr, label):
    if dataset_name == 'pokec_z':
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(-1, 0)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(0, 0)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(1, 0)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(2, 1)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(3, 1)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(4, 1)

    elif dataset_name == 'pokec_n':
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(-1, 0)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(0, 1)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(1, 1)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(2, 1)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(3, 1)

    total_number_of_sens0 = len(df.loc[df[sens_attr] == 0])
    total_number_of_sens1 = len(df.loc[df[sens_attr] == 1])

    number_of_positive_sens0 = len(df.loc[(df[sens_attr] == 0) & (df[label] == 1)])
    number_of_positive_sens1 = len(df.loc[(df[sens_attr] == 1) & (df[label] == 1)])

    fairness = np.absolute(number_of_positive_sens0) / np.absolute(total_number_of_sens0) - np.absolute(number_of_positive_sens1) / np.absolute(total_number_of_sens1)

    return fairness * 100













