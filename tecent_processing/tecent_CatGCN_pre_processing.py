from asyncore import readwrite
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
from fainress_component import disparate_impact_remover, reweighting, sample
import time

def tec_CatGCN_pre_process(df, df_user, df_click, df_item, sens_attr, label, special_case, debaising_approach=None):
    if debaising_approach != None:
        if special_case == True:
            df_user.dropna(inplace=True)
            age_dic = {'11~15':0, '16~20':0, '21~25':0, '26~30':1, '31~35':1, '36~40':2, '41~45':2, '46~50':3, '51~55':3, '56~60':4, '61~65':4, '66~70':4, '71~':4}
            df_user[["age_range"]] = df_user[["age_range"]].applymap(lambda x:age_dic[x])
            df_user.rename(columns={"user_id":"uid", "age_range":"age"}, inplace=True)
            # binarize age
            df_user = apply_bin_age(df_user)
            if debaising_approach == 'disparate_impact_remover':
                df_user = disparate_impact_remover(df_user, sens_attr, label)
            elif debaising_approach == 'reweighting':
                df_user = reweighting(df_user, sens_attr, label)
            elif debaising_approach == 'sample':
                df_user = sample(df_user, sens_attr, label)
        else:
            # binarize age
            df_user = apply_bin_age(df_user)
            df.dropna(inplace=True)
            age_dic = {'11~15':0, '16~20':0, '21~25':0, '26~30':1, '31~35':1, '36~40':2, '41~45':2, '46~50':3, '51~55':3, '56~60':4, '61~65':4, '66~70':4, '71~':4}
            df[["age_range"]] = df[["age_range"]].applymap(lambda x:age_dic[x])
            df.rename(columns={"user_id":"uid","age_range":"age"}, inplace=True)

            df = apply_bin_age(df)

            df.drop(columns=["cid1", "cid2", "cid1_name", "cid2_name ", "cid3_name", "brand_code", "price", "item_name", "seg_name"], inplace=True)

            if debaising_approach == 'disparate_impact_remover':
                df = disparate_impact_remover(df, sens_attr, label)
            elif debaising_approach == 'reweighting':
                df = reweighting(df, sens_attr, label)
            elif debaising_approach == 'sample':
                df = sample(df, sens_attr, label)
            
            df_user, df_item, df_click = divide_data2(df)

    else:
        if special_case == False:
            print('special case is false')
            df_user, df_item, df_click = divide_data(df)
        df_user.dropna(inplace=True)
        age_dic = {'11~15':0, '16~20':0, '21~25':0, '26~30':1, '31~35':1, '36~40':2, '41~45':2, '46~50':3, '51~55':3, '56~60':4, '61~65':4, '66~70':4, '71~':4}
        df_user[["age_range"]] = df_user[["age_range"]].applymap(lambda x:age_dic[x])
        df_user.rename(columns={"user_id":"uid", "age_range":"age"}, inplace=True)

        # binarize age
        df_user = apply_bin_age(df_user)

    # item
    df_item.dropna(inplace=True)
    df_item.rename(columns={"item_id":"pid", "cid3":"cid"}, inplace=True)
    if debaising_approach == None:
        df_item.drop(columns=["cid1", "cid2", "cid1_name", "cid2_name", "cid3_name", "brand_code", "price", "item_name", "seg_name"], inplace=True)
    df_item.reset_index(drop=True, inplace=True)

    df_item = df_item.sample(frac=0.15, random_state=11)
    df_item.reset_index(drop=True, inplace=True)

    # click
    df_click.dropna(inplace=True)

    if debaising_approach == None and special_case == True:
        df_click.rename(columns={"user_id":"uid", "item_id":"pid"}, inplace=True)
    elif debaising_approach != None and special_case == False:
        df_click.rename(columns={"item_id":"pid"}, inplace=True)
    elif debaising_approach != None and special_case == True:
        df_click.rename(columns={"user_id":"uid", "item_id":"pid"}, inplace=True)

    df_click.reset_index(drop=True, inplace=True)

    df_click = df_click.sample(frac=0.15, random_state=11)
    df_click.reset_index(drop=True, inplace=True)

    df_click = df_click[df_click["uid"].isin(df_user["uid"])]
    df_click = df_click[df_click["pid"].isin(df_item["pid"])]

    df_click.drop_duplicates(inplace=True)
    df_click.reset_index(drop=True, inplace=True)

    # filter df_click (item interactions >= 2)
    # Before filtering
    users = set(df_click.uid.tolist())
    items = set(df_click.pid.tolist())

    print('User before filtering {} and items before filtering {}'.format(len(users), len(items)))

    df_click, uid_activity, pid_popularity = filter_triplets(df_click, 'uid', 'pid', min_uc=0, min_sc=2)

    sparsity = 1. * df_click.shape[0] / (uid_activity.shape[0] * pid_popularity.shape[0])

    print("After filtering, there are %d interaction events from %d users and %d items (sparsity: %.4f%%)" % 
        (df_click.shape[0], uid_activity.shape[0], pid_popularity.shape[0], sparsity * 100))

    # After filtering
    users = set(df_click.uid.tolist())
    items = set(df_click.pid.tolist())

    print('Users after filtering {} and items after filtering {}'.format(len(users), len(items)))

    # Click-item merge
    df_click_item = pd.merge(df_click, df_item, how="inner", on="pid")
    raw_click_item = df_click_item.drop("pid", axis=1, inplace=False)
    raw_click_item.drop_duplicates(inplace=True)

    # filter df_click_item (cid interactions >= 2)
    df_click_item, uid_activity, cid_popularity = filter_triplets(raw_click_item, 'uid', 'cid', min_uc=0, min_sc=2)

    sparsity = 1. * df_click_item.shape[0] / (uid_activity.shape[0] * cid_popularity.shape[0])

    print("After filtering, there are %d interacton events from %d users and %d items (sparsity: %.4f%%)" % 
        (df_click_item.shape[0], uid_activity.shape[0], cid_popularity.shape[0], sparsity * 100))

   # uid-uid analysis
    df_click = df_click[df_click["uid"].isin(df_click_item["uid"])]
    df_click_1 = df_click[["uid", "pid"]].copy()
    df_click_1.rename(columns={"uid":"uid1"}, inplace=True)

    df_click_2 = df_click[["uid", "pid"]].copy()
    df_click_2.rename(columns={"uid":"uid2"}, inplace=True)

    df_click1_click2 = pd.merge(df_click_1, df_click_2, how="inner", on="pid")
    #create df_uid_uid
    df_uid_uid = df_click1_click2.drop("pid", axis=1, inplace=False)
    df_uid_uid.drop_duplicates(inplace=True)

    # delete unneeded dataframes
    del df_click_1, df_click_2, df_click1_click2

    # Map
    # Map
    df_label = df_user[df_user["uid"].isin(df_click_item["uid"])]

    if debaising_approach == None and special_case == True:
        uid2id = {num: i for i, num in enumerate(df_label['uid'])}
    elif debaising_approach == 'sample' or debaising_approach == 'reweighting' and special_case == True:
         uid2id = {num: i for i, num in enumerate(df_label['uid'])}
    else:
        uid2id = {num: i for i, num in enumerate(df_click_item['uid'])}
    cid2id = {num: i for i, num in enumerate(pd.unique(df_click_item['cid']))}

    df_label = col_map(df_label, 'uid', uid2id)
    df_label = label_map(df_label, df_label.columns[1:])

    user_edge = df_uid_uid[df_uid_uid['uid1'].isin(df_click_item['uid'])]
    user_edge = user_edge[user_edge['uid2'].isin(df_click_item['uid'])]

    user_edge = col_map(user_edge, 'uid1', uid2id)
    user_edge = col_map(user_edge, 'uid2', uid2id)

    user_field = col_map(df_click_item, 'uid', uid2id)
    user_field = col_map(user_field, 'cid', cid2id)

    if debaising_approach == 'disparate_impact_remover' or debaising_approach == 'sample' or debaising_approach == 'reweighting':
        user_field = user_field.reset_index()
        user_field = user_field.drop(['uid'], axis=1)

        user_field = user_field.rename(columns={"index": "uid"})
        user_field['uid'] = user_field['uid'].astype(str).astype(int)


    # new
    if debaising_approach != None:
        if 'bin_age' not in df_user:
            df_label = df_label.join(df_user['bin_age']) 

    # Save?
    save_path = './'
    user_edge.to_csv(os.path.join(save_path, "user_edge.csv"), index=False)
    user_field.to_csv(os.path.join(save_path, "user_field.csv"), index=False)
    df_label.to_csv(os.path.join(save_path, "user_labels.csv"), index=False)

    df_label[["uid", "age"]].to_csv(os.path.join(save_path, "user_age.csv"), index=False)
    df_label[["uid", "bin_age"]].to_csv(os.path.join(save_path, "user_bin_age.csv"), index=False)
    df_label[["uid", "gender"]].to_csv(os.path.join(save_path, "user_gender.csv"), index=False)
    user_gender = df_label[["uid", "gender"]]

    NUM_FIELD = 10

    np.random.seed(42)

    user_field = field_reader(os.path.join(save_path, "user_field.csv"))

    neighs = get_neighs(user_field)


    if debaising_approach == 'disparate_impact_remover':
        neighs = [x for x in neighs if x.size != 0]

    sample_neighs = []
    for i in range(len(neighs)):
        sample_neighs.append(list(sample_neigh(neighs[i], NUM_FIELD)))
        
    sample_neighs = np.array(sample_neighs)

    np.save(os.path.join(save_path, 'user_field.npy'), sample_neighs)

    user_field_new = sample_neighs

    user_edge_path = './user_edge.csv'
    user_field_new_path = './user_field.npy'
    user_gender_path = './user_gender.csv'
    user_label_path = './user_labels.csv'

    return user_edge_path, user_field_new_path, user_gender_path, user_label_path


def divide_data(df):
    df_user = df[['user_id', 'gender', 'age_range']].copy()
    df_item = df[['item_id', 'cid1', 'cid2', 'cid3', 'cid1_name', 'cid2_name ', 'cid3_name', 'brand_code', 'price', 'item_name', 'seg_name']].copy()
    df_click = df[['user_id', 'item_id']].copy()

    return df_user, df_item, df_click


def divide_data2(df):
    df_user = df[['uid', 'gender', 'age']].copy()
    df_item = df[['item_id', 'cid3']].copy()
    df_click = df[['uid', 'item_id']].copy()

    return df_user, df_item, df_click

def apply_bin_age(df_user):
    df_user["bin_age"] = df_user["age"]
    df_user["bin_age"] = df_user["bin_age"].replace(1,0)
    df_user["bin_age"] = df_user["bin_age"].replace(2,1)
    df_user["bin_age"] = df_user["bin_age"].replace(3,1)
    df_user["bin_age"] = df_user["bin_age"].replace(4,1)

    return df_user

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=True)
    count = playcount_groupbyid.size()
    return count

def filter_triplets(tp, user, item, min_uc=0, min_sc=0):
    # Only keep the triplets for users who clicked on at least min_uc items
    if min_uc > 0:
        usercount = get_count(tp, user)
        tp = tp[tp[user].isin(usercount.index[usercount >= min_uc])]
    
    # Only keep the triplets for items which were clicked on by at least min_sc users. 
    if min_sc > 0:
        itemcount = get_count(tp, item)
        tp = tp[tp[item].isin(itemcount.index[itemcount >= min_sc])]
    
    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, user), get_count(tp, item) 
    return tp, usercount, itemcount

def col_map(df, col, num2id):
    df[[col]] = df[[col]].applymap(lambda x: num2id[x])
    return df


def label_map(label_df, label_list):
    for label in label_list:
        label2id = {num: i for i, num in enumerate(pd.unique(label_df[label]))}
        label_df = col_map(label_df, label, label2id)
    return label_df

def field_reader(path):
    """
    Reading the sparse field matrix stored as csv from the disk.
    :param path: Path to the csv file.
    :return field: csr matrix of field.
    """
    user_field = pd.read_csv(path)
    user_index = user_field["uid"].values.tolist()
    field_index = user_field["cid"].values.tolist()
    user_count = max(user_index)+1
    field_count = max(field_index)+1
    field_index = sp.csr_matrix((np.ones_like(user_index), (user_index, field_index)), shape=(user_count, field_count))
    return field_index

def get_neighs(csr):
    neighs = []
#     t = time.time()
    idx = np.arange(csr.shape[1])
    for i in range(csr.shape[0]):
        x = csr[i, :].toarray()[0] > 0
        neighs.append(idx[x])
#         if i % (10*1000) == 0:
#             print('sec/10k:', time.time()-t)
    return neighs

def sample_neigh(neigh, num_sample):
    if len(neigh) >= num_sample:
        sample_neigh = np.random.choice(neigh, num_sample, replace=False)
    elif len(neigh) < num_sample:
        sample_neigh = np.random.choice(neigh, num_sample, replace=True)
    return sample_neigh