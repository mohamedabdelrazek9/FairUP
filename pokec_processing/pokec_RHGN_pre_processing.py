from turtle import pd
import numpy as np
import pandas as pd
import dgl
from fainress_component import disparate_impact_remover, reweighting, sample
import fastText
import torch

def pokec_z_RHGN_pre_process(df, dataset_user_id_name, sens_attr, label, debaising_approach=None):

    
    df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(-1, 0)
    #df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(0, 0)
    df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(1, 0)
    df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(2, 1)
    df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(3, 1)
    df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(4, 1)

    if debaising_approach != 'sample':
        df = df.astype({'user_id': 'str'}, copy=False)
        df = df.astype({'completion_percentage':'str', 'AGE':'str', 'I_am_working_in_field':'str'}, copy=False)

    if debaising_approach != None:
        if debaising_approach == 'disparate_impact_remover':
            df = disparate_impact_remover(df, sens_attr, label)
        elif debaising_approach == 'reweighting':
            df = reweighting(df, sens_attr, label)
        elif debaising_approach == 'sample':
            df = sample(df, sens_attr, label)
            df = df.astype({'user_id':'str'}, copy=False)
            df = df.astype({'completion_percentage':'str', 'AGE':'str', 'I_am_working_in_field':'str'}, copy=False)

    if debaising_approach == 'reweighting' or debaising_approach == 'disparate_impact_remover':
        df.user_id = df.user_id.astype(np.int64)
        df.user_id = df.user_id.astype(str)

        df.completion_percentage = df.completion_percentage.astype(np.int64)
        df.completion_percentage = df.completion_percentage.astype(str)

        df.AGE = df.AGE.astype(np.int64)
        df.AGE = df.AGE.astype(str)

        df.I_am_working_in_field = df.I_am_working_in_field.astype(np.int64)
        df.I_am_working_in_field = df.I_am_working_in_field.astype(str)

    
    user_dic = {k: v for v, k in enumerate(df.user_id.drop_duplicates())}
    comp_dic = {k: v for v, k in enumerate(df.completion_percentage.drop_duplicates())}
    age_dic = {k: v for v, k in enumerate(df.AGE.drop_duplicates())}
    working_dic = {k: v for v, k in enumerate(df.I_am_working_in_field.drop_duplicates())}

    item_dic = {}
    c1, c2, c3=[], [], []
    '''
    if debaising_approach == 'sample':
        for i, row in df.iterrows():
            c1_1 = df.at[i, 'completion_percentage']
            if isinstance(c1_1, str):
                c1.append(comp_dic[c1_1])
            else:
                c1.append(comp_dic[c1_1.iloc[0]])

            c2_2 = df.at[i, 'AGE']
            if isinstance(c2_2, str):
                c2.append(age_dic[c2_2])
            else:
                c2.append(age_dic[c2_2.iloc[0]])

            c3_3 = df.at[i, 'I_am_working_in_field']
            if isinstance(c3_3, str):
                c3.append(working_dic[c3_3])
            else:
                c3.append(working_dic[c3_3.iloc[0]])
    '''
    if debaising_approach == 'disparate_impact_remover' or debaising_approach == 'reweighting':
        for i in range(len(df)):
            c1.append(comp_dic[df['completion_percentage'].iloc[i]])
            c2.append(age_dic[df['AGE'].iloc[i]])
            c3.append(working_dic[df['I_am_working_in_field'].iloc[i]])
    else:
        for i in range(len(df)):
            c1.append(comp_dic[df.at[i, 'completion_percentage']])
            c2.append(age_dic[df.at[i, 'AGE']])
            c3.append(working_dic[df.at[i, 'I_am_working_in_field']])
        
        
    print(min(c1), min(c2), min(c3))
    print(len(comp_dic), len(age_dic), len(working_dic))

    has_user = [user_dic[user] for user in df.user_id]
    is_made_by_user = [age_dic[item] for item in df.AGE]


    data_dict = {
        ("user", "has", "item"): (torch.tensor(has_user), torch.tensor(is_made_by_user)),
        ("item", "is_made_by", "user"): (torch.tensor(is_made_by_user), torch.tensor(has_user))
    }

    G = dgl.heterograph(data_dict)

    model = fasttext.load_model('../cc.zh.200.bin')

    temp1 = {k: model.get_sentence_vector(v) for v, k in comp_dic.items()}
    cid1_feature = torch.tensor([temp1[k] for _, k in comp_dic.items()])

    temp2 = {k: model.get_sentence_vector(v) for v, k in age_dic.items()}
    cid2_feature = torch.tensor([temp2[k] for _, k in age_dic.items()])
 

    temp3 = {k: model.get_sentence_vector(v) for v, k in working_dic.items()}
    cid3_feature = torch.tensor([temp3[k] for _, k in working_dic.items()])

    uid2id = {num: i for i, num in enumerate(df[dataset_user_id_name])}

    df_user = col_map(df, dataset_user_id_name, uid2id)
    user_label = label_map(df_user, df_user.columns[1:])

    label_age = user_label.AGE
    label_comp_perc = user_label.completion_percentage
    label_gender = user_label.gender
    label_region = user_label.region
    label_working = user_label.I_am_working_in_field
    label_lang = user_label.spoken_languages_indicator


    G.nodes['user'].data['age'] = torch.tensor(label_age[:G.number_of_nodes('user')])
    G.nodes['user'].data['completion_percentage'] = torch.tensor(label_comp_perc[:G.number_of_nodes('user')])
    G.nodes['user'].data['gender'] = torch.tensor(label_gender[:G.number_of_nodes('user')])
    G.nodes['user'].data['region'] = torch.tensor(label_region[:G.number_of_nodes('user')])
    G.nodes['user'].data['I_am_working_in_field'] = torch.tensor(label_working[:G.number_of_nodes('user')])
    G.nodes['user'].data['spoken_languages_indicator'] = torch.tensor(label_lang[:G.number_of_nodes('user')])

    G.nodes['item'].data['cid1'] = torch.tensor(c1[:G.number_of_nodes('item')])
    G.nodes['item'].data['cid2'] = torch.tensor(c2[:G.number_of_nodes('item')])
    G.nodes['item'].data['cid3'] = torch.tensor(c3[:G.number_of_nodes('item')])

    
    print(G)
    print(cid1_feature.shape)
    print(cid2_feature.shape)
    print(cid3_feature.shape)

    return G, cid1_feature, cid2_feature, cid3_feature


def col_map(df, col, num2id):
    df[[col]] = df[[col]].applymap(lambda x: num2id[x])
    return df

def label_map(label_df, label_list):
    for label in label_list:
        label2id = {num: i for i, num in enumerate(pd.unique(label_df[label]))}
        label_df = col_map(label_df, label, label2id)
    return label_df