from dis import dis
import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
from fainress_component import disparate_impact_remover, reweighting, sample

def pokec_z_CatGCN_pre_process(df, df_edge_list, sens_attr, label, debaising_approach=True):

    df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(-1, 0)
    df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(0, 0)
    df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(1, 0)
    df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(2, 1)
    df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(3, 1)
    df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(4, 1)

    if debaising_approach != None:
        if debaising_approach == 'disparate_impact_remover':
            df = disparate_impact_remover(df, sens_attr, label)
        elif debaising_approach == 'reweighting':
            df = reweighting(df, sens_attr, label)
        elif debaising_approach == 'sample':
            df = sample(df, sens_attr, label)

    uid_age = df[['user_id', 'AGE']].copy()
    uid_age.dropna(inplace=True)
    uid_age2 = df[['user_id', 'AGE']].copy()

    #create uid2id
    uid2id = {num: i for i, num in enumerate(df['user_id'])}
    #create age2id
    age2id = {num: i for i, num in enumerate(pd.unique(uid_age['AGE']))}

    #create user_field
    user_field = col_map(uid_age, 'user_id', uid2id)
    user_field = col_map(user_field, 'AGE', age2id)


    if debaising_approach == 'disparate_impact_remover':
        user_field = user_field.reset_index()
        user_field = user_field.drop(['user_id'], axis=1)

        user_field = user_field.rename(columns={"index": "user_id"})
        user_field['user_id'] = user_field['user_id'].astype(str).astype(int)

    #create user_label
    user_label = df[df['user_id'].isin(uid_age2['user_id'])]
    user_label = col_map(user_label, 'user_id', uid2id)
    user_label = label_map(user_label, user_label.columns[1:])

    # save_path = "./input_ali_data"
    save_path = "./"
    # process edge list
    #if df_edge_list['source'].dtype != 'int64':
    #    df_edge_list['source'] = df_edge_list['source'].astype(str).astype(np.int64)
    #    df_edge_list['target'] = df_edge_list['target'].astype(str).astype(np.int64)

    source = []
    target = []
    print('adjusting edge list')
    #for i in range(df_edge_list.shape[0]):
    #    print(i)
    #    if any(df.user_id == df_edge_list.source[i]) == True and any(df.user_id == df_edge_list.target[i]) == True:
    #        index = df.user_id[df.user_id == df_edge_list.source[i]].index.tolist()[0]
    #        source.append(index)
    #        index2 = df.user_id[df.user_id == df_edge_list.target[i]].index.tolist()[0]
    #        target.append(index2)

    #user_edge_new = pd.DataFrame({'uid': source, 'uid2': target})
    print('saving edge list')
    user_edge_new = df_edge_list
    user_edge_new.to_csv(os.path.join(save_path, 'user_edge.csv'), index=False)
    user_field.to_csv(os.path.join(save_path, 'user_field.csv'), index=False)
    user_label.to_csv(os.path.join(save_path, 'user_labels.csv'), index=False)

    user_label[['user_id','public']].to_csv(os.path.join(save_path, 'user_public.csv'), index=False)
    user_label[['user_id','completion_percentage']].to_csv(os.path.join(save_path, 'user_completion_percentage.csv'), index=False)
    user_label[['user_id','gender']].to_csv(os.path.join(save_path, 'user_gender.csv'), index=False)
    user_label[['user_id','region']].to_csv(os.path.join(save_path, 'user_region.csv'), index=False)
    user_label[['user_id','AGE']].to_csv(os.path.join(save_path, 'user_age.csv'), index=False)
    user_label[['user_id','I_am_working_in_field']].to_csv(os.path.join(save_path, 'user_work.csv'), index=False)
    user_work = user_label[['user_id','I_am_working_in_field']]
    user_label[['user_id','spoken_languages_indicator']].to_csv(os.path.join(save_path, 'user_spoken_languages_indicator.csv'), index=False)

    NUM_FIELD = 10
    #np.random_seed(42)

     # load user_field.csv
    user_field = field_reader(os.path.join(save_path, 'user_field.csv'))
    print("Shapes of user with field:", user_field.shape)
    print("Number of user with field:", np.count_nonzero(np.sum(user_field, axis=1)))

    neighs = get_neighs(user_field)

    sample_neighs = []
    for i in range(len(neighs)):
        sample_neighs.append(list(sample_neigh(neighs[i], NUM_FIELD)))
    sample_neighs = np.array(sample_neighs)

    np.save(os.path.join(save_path, 'user_field.npy'), sample_neighs)

    user_field_new = sample_neighs

    user_edge_path = './user_edge.csv'
    user_field_new_path = './user_field.npy'
    user_work_path = './user_work.csv'
    user_label_path = './user_labels.csv'

    return user_edge_path, user_field_new_path, user_work_path, user_label_path


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
    user_index = user_field["user_id"].values.tolist()
    field_index = user_field["AGE"].values.tolist()
    user_count = max(user_index)+1
    field_count = max(field_index)+1
    field_index = sp.csr_matrix((np.ones_like(user_index), (user_index, field_index)), shape=(user_count, field_count))
    return field_index

#user_field = field_reader(os.path.join(save_path, 'user_field.csv'))

#print("Shapes of user with field:", user_field.shape)
#print("Number of user with field:", np.count_nonzero(np.sum(user_field, axis=1)))

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