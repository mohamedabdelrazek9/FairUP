import pandas as pd
import numpy as np
import torch
import dgl
import fastText
from fainress_component import disparate_impact_remover, reweighting, sample

def tec_RHGN_pre_process(df, df_user, df_click, df_item, sens_attr, label, special_case, debaising_approach=None):
    # load and clean data
    if debaising_approach != None:
        if special_case == True:
            df_user.dropna(inplace=True)
            age_dic = {'11~15':0, '16~20':0, '21~25':0, '26~30':1, '31~35':1, '36~40':2, '41~45':2, '46~50':3, '51~55':3, '56~60':4, '61~65':4, '66~70':4, '71~':4}
            df_user[["age_range"]] = df_user[["age_range"]].applymap(lambda x:age_dic[x])
            df_user.rename(columns={"user_id":"uid", "age_range":"age"}, inplace=True)

            # binarize age
            df_user = apply_bin_age(df_user)
            #df_extra = df[['cid1_name', 'cid2_name ', 'cid3_name']].copy()
            #df.drop(columns=["cid1_name", "cid2_name ", "cid3_name", "item_name", "seg_name"], inplace=True)
            if debaising_approach == 'disparate_impact_remover':
                df_user = disparate_impact_remover(df_user, sens_attr, label)
            elif debaising_approach == 'reweighting':
                df_user = reweighting(df_user, sens_attr, label)
            elif debaising_approach == 'sample':
                df_user = sample(df_user, sens_attr, label)

        else:
            df.dropna(inplace=True)

            age_dic = {'11~15':0, '16~20':0, '21~25':0, '26~30':1, '31~35':1, '36~40':2, '41~45':2, '46~50':3, '51~55':3, '56~60':4, '61~65':4, '66~70':4, '71~':4}
            df[["age_range"]] = df[["age_range"]].applymap(lambda x:age_dic[x])
            df.rename(columns={"user_id":"uid", "age_range":"age"}, inplace=True)

            df = apply_bin_age(df)
            df_extra = df[['cid1_name', 'cid2_name ', 'cid3_name']].copy()
            df.drop(columns=["cid1_name", "cid2_name ", "cid3_name", "item_name", "seg_name"], inplace=True)

            if debaising_approach == 'disparate_impact_remover':
                df = disparate_impact_remover(df, sens_attr, label)
            elif debaising_approach == 'reweighting':
                df = reweighting(df, sens_attr, label)
            elif debaising_approach == 'sample':
                df = sample(df, sens_attr, label)

            df_user, df_item, df_click = divide_data2(df)

    else:
        if special_case == False:
            print('special case is False')
            df_user, df_item, df_click = divide_data(df)

        # df_user process
        df_user.dropna(inplace=True)
        age_dic = {'11~15':0, '16~20':0, '21~25':0, '26~30':1, '31~35':1, '36~40':2, '41~45':2, '46~50':3, '51~55':3, '56~60':4, '61~65':4, '66~70':4, '71~':4}
        df_user[["age_range"]] = df_user[["age_range"]].applymap(lambda x:age_dic[x])
        df_user.rename(columns={"user_id":"uid", "age_range":"age"}, inplace=True)

        # binarize age
        df_user = apply_bin_age(df_user)

    # df_item process
    df_item.dropna(inplace=True)
    df_item.rename(columns={"item_id":"pid", "brand_code":"brand"}, inplace=True)
    df_item.reset_index(drop=True, inplace=True)

    df_item = df_item.sample(frac=0.15, random_state=11)
    df_item.reset_index(drop=True, inplace=True)

    # df_click process
    df_click.dropna(inplace=True)
    df_click.rename(columns={"user_id":"uid", "item_id":"pid"}, inplace=True)
    df_click.reset_index(drop=True, inplace=True)

    df_click = df_click.sample(frac=0.15, random_state=11)
    df_click.reset_index(drop=True, inplace=True)
    
    df_click = df_click[df_click["uid"].isin(df_user["uid"])]
    df_click = df_click[df_click["pid"].isin(df_item["pid"])]

    df_click.drop_duplicates(inplace=True)
    df_click.reset_index(drop=True, inplace=True)

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

    # Process
    df_user = df_user[df_user['uid'].isin(users)]
    df_item = df_item[df_item['pid'].isin(items)]
    df_user.reset_index(drop=True, inplace=True)
    df_item.reset_index(drop=True, inplace=True)

    df_user = df_user.astype({"uid": "str"}, copy=False)
    df_item = df_item.astype({'pid': 'str', 'cid1': 'str', 'cid2': 'str', 'cid3': 'str', 'brand': 'str'}, copy=False)
    df_click = df_click.astype({'uid': 'str', 'pid': 'str'}, copy=False)

    if debaising_approach != None and special_case == True:
        df_user.uid = df_user.uid.astype(float).astype(int)  # works
        df_user.uid = df_user.uid.astype(str)

    # Build a dictionary and remove duplicate items
    if debaising_approach != None and special_case == False:
        user_dic = {k: v for v,k in enumerate(df_user.uid)}
        cid1_dic = {k: v for v,k in enumerate(df_extra.cid1_name.drop_duplicates())}
        cid2_dic = {k: v for v,k in enumerate(df_extra['cid2_name'].drop_duplicates())}
        cid3_dic = {k: v for v,k in enumerate(df_extra.cid3_name.drop_duplicates())}
        brand_dic = {k: v for v, k in enumerate(df_item.brand.drop_duplicates())}
    else:
        user_dic = {k: v for v,k in enumerate(df_user.uid)}
        cid1_dic = {k: v for v, k in enumerate(df_item.cid1_name.drop_duplicates())}  
        cid2_dic = {k: v for v, k in enumerate(df_item['cid2_name'].drop_duplicates())}
        cid3_dic = {k: v for v, k in enumerate(df_item.cid3_name.drop_duplicates())}
        brand_dic = {k: v for v, k in enumerate(df_item.brand.drop_duplicates())}
    item_dic = {}
    c1, c2, c3, brand = [], [], [], []
    for i in range(len(df_item)):
        k = df_item.at[i,'pid']
        v = i
        item_dic[k] = v
        if debaising_approach != None and special_case == False:
            c1.append(cid1_dic[df_extra.at[i,'cid1_name']])
            c2.append(cid2_dic[df_extra.at[i,'cid2_name']])
            c3.append(cid3_dic[df_extra.at[i,'cid3_name']])
            brand.append(brand_dic[df_item.at[i,'brand']])
        else:
            c1.append(cid1_dic[df_item.at[i,'cid1_name']])
            c2.append(cid2_dic[df_item.at[i,'cid2_name']])
            c3.append(cid3_dic[df_item.at[i,'cid3_name']])
            brand.append(brand_dic[df_item.at[i,'brand']])

    if debaising_approach != None:
        df_item.drop(columns=["price"], inplace=True)
    else:
        df_item.drop(columns=["cid1_name", "cid2_name", "cid3_name", "price", "item_name", "seg_name"], inplace=True)

    #df_user['bin_age'] = df_user['bin_age'].replace(1,2)
    #df_user['bin_age'] = df_user['bin_age'].replace(0,1)
    #df_user['bin_age'] = df_user['bin_age'].replace(2,0)

    if debaising_approach != None:
        if 'bin_age' not in df_user:
            df_user = df_user.join(df_user['bin_age']) 

    # Save?

    # Generate Graph
    G, cid1_feature, cid2_feature, cid3_feature, brand_feature = generate_graph(df_user, 
                                                                                df_item, 
                                                                                df_click, 
                                                                                user_dic, 
                                                                                item_dic, 
                                                                                cid1_dic, 
                                                                                cid2_dic, 
                                                                                cid3_dic, 
                                                                                brand_dic, 
                                                                                c1, 
                                                                                c2, 
                                                                                c3, 
                                                                                brand,
                                                                                debaising_approach)
    

    return G, cid1_feature, cid2_feature, cid3_feature, brand_feature # brand_feature not used (same as cid4_feature?)


def divide_data(df):
    df_user = df[['user_id', 'gender', 'age_range']].copy()
    df_item = df[['item_id', 'cid1', 'cid2', 'cid3', 'cid1_name', 'cid2_name', 'cid3_name','brand_code', 'price', 'item_name', 'seg_name']].copy()
    df_click = df[['user_id', 'item_id']].copy()

    return df_user, df_item, df_click

def divide_data2(df):
    df_user = df[['uid', 'gender', 'age']].copy()
    df_item = df[['item_id', 'cid1', 'cid2', 'cid3', 'brand_code', 'price']].copy()
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


def generate_graph(df_user, df_item, df_click, user_dic, item_dic, cid1_dic, cid2_dic, cid3_dic, brand_dic, c1, c2, c3, brand, debaising_approach):

    u = {v:k for k,v in user_dic.items()}
    i = {v:k for k,v in item_dic.items()}

    click_user = [user_dic[user] for user in df_click.uid]
    click_item = [item_dic[item] for item in df_click.pid]

    data_dict = {
        ('user', 'click', 'item'): (torch.tensor(click_user), torch.tensor(click_item)),
        ('item', 'click-by', 'user'): (torch.tensor(click_item), torch.tensor(click_user))
    }

    
    G = dgl.heterograph(data_dict)

    # todo import the fasttext correctly
    model = fasttext.load_model('../cc.zh.200.bin')

    temp = {k: model.get_sentence_vector(v) for v, k in cid1_dic.items()}
    cid1_feature = torch.tensor([temp[k] for _, k in cid1_dic.items()])

    temp = {k: model.get_sentence_vector(v) for v, k in cid2_dic.items()}
    cid2_feature = torch.tensor([temp[k] for _, k in cid2_dic.items()])

    temp = {k: model.get_sentence_vector(v) for v, k in cid3_dic.items()}
    cid3_feature = torch.tensor([temp[k] for _, k in cid3_dic.items()])

    temp = {k: model.get_sentence_vector(v) for v, k in brand_dic.items()}
    brand_feature = torch.tensor([temp[k] for _, k in brand_dic.items()])

    # Passing labels into label
    if debaising_approach == 'disparate_impact_remover' or debaising_approach == 'reweighting':
        df_user['gender'] = df_user['gender'].astype(np.int64)
    label_gender = df_user.gender
    label_age = df_user.age
    label_bin_age = df_user.bin_age

    G.nodes['user'].data['gender'] = torch.tensor(label_gender[:G.number_of_nodes('user')])
    G.nodes['user'].data['age'] = torch.tensor(label_age[:G.number_of_nodes('user')])
    G.nodes['user'].data['bin_age'] = torch.tensor(label_bin_age[:G.number_of_nodes('user')])
    G.nodes['item'].data['cid1'] = torch.tensor(c1[:G.number_of_nodes('item')])
    G.nodes['item'].data['cid2'] = torch.tensor(c2[:G.number_of_nodes('item')])
    G.nodes['item'].data['cid3'] = torch.tensor(c3[:G.number_of_nodes('item')])
    G.nodes['item'].data['brand'] = torch.tensor(brand[:G.number_of_nodes('item')])

    return G, cid1_feature, cid2_feature, cid3_feature, brand_feature