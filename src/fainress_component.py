from cProfile import label
import numpy as np
import pandas as pd
import networkx as nx
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover, Reweighing, LFR
from aif360.metrics import BinaryLabelDatasetMetric


def fairness_calculation(dataset_name, sens_attr, predict_attr):

    if dataset_name == 'nba':
        dataset_path = '/datasets/NBA/nba.csv'
        fairness_calculation_nba(dataset_path, sens_attr, predict_attr)
    
    elif dataset_name == 'alibaba':
        dataset_path = '/datasets/NBA/nba.csv'
        fairness_calculation_alibaba(dataset_path, sens_attr, predict_attr)

    elif dataset_name == 'tecent':
        dataset_path = '/datasets/NBA/nba.csv'
        fairness_calculation_tecent(dataset_path, sens_attr, predict_attr)

    elif dataset_name == 'pokec_z' or dataset_name == 'pokec_n':
        dataset_path = '/datasets/NBA/nba.csv'
        fairness_calculation_pokec(dataset_path, dataset_path, sens_attr, predict_attr)


def fairness_calculation_nba(dataset_path, sens_attr, predict_attr):
    #data = nx.read_graphml(dataset_path)
    #df = pd.DataFrame.from_dict(dict(data.nodes(data=True)), orient='index')
    df = pd.read_csv(dataset_path)

    if df.columns[0] != 'user_id':
        df = df.reset_index(level=0)
        df = df.rename(columns={"index": "user_id"})

    if type(df['user_id'][0]) != np.int64:
        df['user_id'] = pd.to_numeric(df['user_id'])
        df = df.astype({'user_id': int})

    df[predict_attr] = df[predict_attr].replace(-1, 0)

    #dataset_fairness(df, sens_attr, predict_attr)

    disparate_impact(df, sens_attr, predict_attr)

def fairness_calculation_alibaba(dataset_path, sens_attr, label):
   # data = nx.read_graphml(dataset_path)
    #df = pd.DataFrame.from_dict(dict(data.nodes(data=True)), orient='index')
    df = pd.read_csv(dataset_path)

    #if df.columns[0] != 'userid':
    #    df = df.reset_index(level=0)
    #    df = df.rename(columns={"index": "userid"})

    #if type(df['userid'][0]) != np.int64:
    #    df['userid'] = pd.to_numeric(df['userid'])
    #    df = df.astype({'userid': int})

    #if sens_attr == 'age' or sens_attr == 'age_level' or sens_attr == 'bin_age':
    #    df.rename(columns={'age_level':'age', 'final_gender_code':'gender'}, inplace=True)

    df[sens_attr] = df[sens_attr].replace(1, 0)
    df[sens_attr] = df[sens_attr].replace(2, 0)
    df[sens_attr] = df[sens_attr].replace(3, 0)
    df[sens_attr] = df[sens_attr].replace(4, 1)
    df[sens_attr] = df[sens_attr].replace(5, 1)
    df[sens_attr] = df[sens_attr].replace(6, 1)

    df[label] = df[label].replace(1, 0)
    df[label] = df[label].replace(2, 1)

    #dataset_fairness(df, sens_attr, label)

    disparate_impact(df, sens_attr, label)

def fairness_calculation_tecent(dataset_path, sens_attr, label):
    #data = nx.read_graphml(dataset_path)
    #df = pd.DataFrame.from_dict(dict(data.nodes(data=True)), orient='index')
    df = pd.read_csv(dataset_path)

    #if df.columns[0] != 'user_id':
    #    df = df.reset_index(level=0)
     #   df = df.rename(columns={"index": "user_id"})

    #if type(df['user_id'][0]) != np.int64:
     #   df['user_id'] = pd.to_numeric(df['user_id'])
     #   df = df.astype({'user_id': int})

    #if sens_attr == 'bin_age':
    #    df.rename(columns={'age_range':'age'}, inplace=True)

    if sens_attr == 'age_range':
        age_dic = {'11~15':0, '16~20':0, '21~25':0, '26~30':1, '31~35':1, '36~40':2, '41~45':2, '46~50':3, '51~55':3, '56~60':4, '61~65':4, '66~70':4, '71~':4}
        df[[sens_attr]] = df[[sens_attr]].applymap(lambda x:age_dic[x])

        df[sens_attr] = df[sens_attr].replace(1,0)
        df[sens_attr] = df[sens_attr].replace(2,1)
        df[sens_attr] = df[sens_attr].replace(3,1)
        df[sens_attr] = df[sens_attr].replace(4,1)

    #dataset_fairness(df, sens_attr, label)

    disparate_impact(df, sens_attr, label)
    
def fairness_calculation_pokec(dataset_path, dataset_name, sens_attr, label):
    #data = nx.read_graphml(dataset_path)
    #df = pd.DataFrame.from_dict(dict(data.nodes(data=True)), orient='index')
    df = pd.read_csv(dataset_path)

    #if df.columns[0] != 'user_id':
    #    df = df.reset_index(level=0)
    #    df = df.rename(columns={"index": "user_id"})

    #if type(df['user_id'][0]) != np.int64:
    #    df['user_id'] = pd.to_numeric(df['user_id'])
    #    df = df.astype({'user_id': int})

    if dataset_name == 'pokec_z':
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(-1, 0)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(0, 0)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(1, 0)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(2, 1)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(3, 1)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(4, 1)

    #elif dataset_name == 'pokec_n':
    #    df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(-1, 0)
    #    df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(0, 1)
    #    df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(1, 1)
    #    df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(2, 1)
    #    df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(3, 1)

    #dataset_fairness(df, sens_attr, label)

    disparate_impact(df, sens_attr, label)

def dataset_fairness(df, sens_attr, label):
    total_number_of_sens0 = len(df.loc[df[sens_attr] == 0])
    total_number_of_sens1 = len(df.loc[df[sens_attr] == 1])

    number_of_positive_sens0 = len(df.loc[(df[sens_attr] == 0) & (df[label] == 1)])
    number_of_positive_sens1 = len(df.loc[(df[sens_attr] == 1) & (df[label] == 1)])

    fairness = np.absolute(number_of_positive_sens0) / np.absolute(total_number_of_sens0) - np.absolute(number_of_positive_sens1) / np.absolute(total_number_of_sens1)
    dataset_fainress = fairness * 100

    print('Dataset fairness:', dataset_fainress)


def disparate_impact(df, sens_attr, label):

    pr_unpriv = calc_prop(df, sens_attr, 1, label, 1)
    #print('pr_unpriv: ', pr_unpriv)

    pr_priv = calc_prop(df, sens_attr, 0, label, 1)
    #print('pr_priv:', pr_priv)
    disp = pr_unpriv / pr_priv

    bin_label_dataset = BinaryLabelDataset(favorable_label=1, 
                                           unfavorable_label=0, 
                                           df=df, 
                                           label_names=[label], 
                                           protected_attribute_names=[sens_attr], 
                                           unprivileged_protected_attributes=[1])

    privileged_groups = [{sens_attr: 0}] 
    unprivileged_groups = [{sens_attr: 1}] 
    metric_dataset = BinaryLabelDatasetMetric(bin_label_dataset, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    # just for comparison
    print('Dataset Fairness:', disp)
    #print("Disparate impact (from AIF360) = %f" %metric_dataset.disparate_impact()) 


def calc_prop(data, group_col, group, output_col, output_val):
    new = data[data[group_col] == group]
    return len(new[new[output_col] == output_val])/len(new)


def disparate_impact_remover(df, sens_attr, label):

    if 'final_gender_code' in df:
        df.rename(columns={'final_gender_code':'gender'}, inplace=True)

    elif 'age_level' in df:
        df.rename(columns={'age_level': 'age'}, inplace=True)

    bin_label_dataset = BinaryLabelDataset(favorable_label=1, 
                                           unfavorable_label=0, 
                                           df=df, 
                                           label_names=[label], 
                                           protected_attribute_names=[sens_attr], 
                                           unprivileged_protected_attributes=[1])

    di = DisparateImpactRemover(repair_level=1    )
    di_transformation = di.fit_transform(bin_label_dataset)

    privileged_groups = [{sens_attr: 0}] 
    unprivileged_groups = [{sens_attr: 1}] 

    metric_original_dataset = BinaryLabelDatasetMetric(bin_label_dataset, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    metric_new_dataset = BinaryLabelDatasetMetric(di_transformation, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    print("Original Disparate impact (from AIF360) = %f" %metric_original_dataset.disparate_impact())
    print("After debaising Disparate impact (from AIF360) = %f" %metric_new_dataset.disparate_impact())

    new_df = di_transformation.convert_to_dataframe()[0]

    return new_df


def reweighting(df, sens_attr, label):
    print('we are in reweighting')

    bin_label_dataset = BinaryLabelDataset(favorable_label=1, 
                                           unfavorable_label=0, 
                                           df=df, 
                                           label_names=[label], 
                                           protected_attribute_names=[sens_attr], 
                                           unprivileged_protected_attributes=[1])

    privileged_groups = [{sens_attr: 0}] 
    unprivileged_groups = [{sens_attr: 1}] 

    RW = Reweighing(unprivileged_groups = unprivileged_groups, privileged_groups   = privileged_groups)

    RW.fit(bin_label_dataset)
    rw_transformation = RW.transform(bin_label_dataset)

    metric_original_dataset = BinaryLabelDatasetMetric(bin_label_dataset, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    metric_new_dataset = BinaryLabelDatasetMetric(rw_transformation, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    print("Original Disparate impact (from AIF360) = %f" %metric_original_dataset.disparate_impact())
    print("After debaising Disparate impact (from AIF360) = %f" %metric_new_dataset.disparate_impact())


    df_new = rw_transformation.convert_to_dataframe()[0]


    return df_new

def lfr(df, sens_attr, label):

    bin_label_dataset = BinaryLabelDataset(favorable_label=1, 
                                           unfavorable_label=0, 
                                           df=df, 
                                           label_names=[label], 
                                           protected_attribute_names=[sens_attr], 
                                           unprivileged_protected_attributes=[1])

    privileged_groups = [{sens_attr: 0}] 
    unprivileged_groups = [{sens_attr: 1}]

    TR = LFR(unprivileged_groups = unprivileged_groups, privileged_groups = privileged_groups)

    TR = TR.fit(bin_label_dataset)

    dset_lfr_trn = TR.transform(bin_label_dataset, threshold = 0.3)
    dset_lfr_trn = bin_label_dataset.align_datasets(dset_lfr_trn)

    metric_original_dataset = BinaryLabelDatasetMetric(bin_label_dataset, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    metric_new_dataset = BinaryLabelDatasetMetric(dset_lfr_trn, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    print("Original Disparate impact (from AIF360) = %f" %metric_original_dataset.disparate_impact())
    print("After debaising Disparate impact (from AIF360) = %f" %metric_new_dataset.disparate_impact())

    df_new = dset_lfr_trn.convert_to_dataframe()[0]

    return df_new

def sample(df, sens_attr, label):
    print('we are in sample')
    dp = df.loc[(df[sens_attr] == 0) & (df[label] == 1)]
    dn = df.loc[(df[sens_attr] == 0) & (df[label] == 0)]
    fp = df.loc[(df[sens_attr] == 1) & (df[label] == 1)]
    fn = df.loc[(df[sens_attr] == 1) & (df[label] == 0)]


    wdp = len(df.loc[df[sens_attr] == 0]) * len(df.loc[df[label] == 1]) / len(df.loc[(df[label] == 1) & (df[sens_attr] == 0)])
    wdn = len(df.loc[df[sens_attr] == 0]) * len(df.loc[df[label] == 0]) / len(df.loc[(df[label] == 1) & (df[sens_attr] == 0)])
    wfp = len(df.loc[df[sens_attr] == 1]) * len(df.loc[df[label] == 1]) / len(df.loc[(df[label] == 1) & (df[sens_attr] == 0)])
    wfn = len(df.loc[df[sens_attr] == 1]) * len(df.loc[df[label] == 0]) / len(df.loc[(df[label] == 1) & (df[sens_attr] == 0)])

    # sample
    dp_sample = dp.sample(n=int(wdp), random_state=1, replace=True)
    dn_sample = dn.sample(n=int(wdn), random_state=1, replace=True)
    fp_sample = fp.sample(n=int(wfp), random_state=1, replace=True)
    fn_sample = fn.sample(n=int(wfn), random_state=1, replace=True)

    # merge
    df_new = pd.concat([dp_sample, dn_sample, fp_sample, fn_sample]).drop_duplicates().reset_index(drop=True)

    return df_new




'''
def fairness_calculation(dataset_path, dataset_name, sens_attr, predict_attr, label):

    data = nx.read_graphml(dataset_path)
    df = pd.DataFrame.from_dict(dict(data.nodes(data=True)), orient='index')

    if df.columns[0] != 'userid':    
        # if so, then we make it as the first column
        df = df.reset_index(level=0)
        df = df.rename(columns={"index": 'userid'})

    # check if user_id column is not string
    if type(df['userid'][0]) != np.int64:
        # if so, we convert it to int
        df['userid'] = pd.to_numeric(df['userid'])
        df = df.astype({'userid': int})

    if predict_attr != None:
        label == predict_attr

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

    elif dataset_name == 'alibaba':
        df['age_level'] = df['age_level'].replace(1, 0)
        df['age_level'] = df['age_level'].replace(2, 0)
        df['age_level'] = df['age_level'].replace(3, 0)
        df['age_level'] = df['age_level'].replace(4, 1)
        df['age_level'] = df['age_level'].replace(5, 1)
        df['age_level'] = df['age_level'].replace(6, 1)

        df['final_gender_code'] = df['final_gender_code'].replace(1, 0)
        df['final_gender_code'] = df['final_gender_code'].replace(2, 1)

        #df.rename(columns={'age_level':'age', 'final_gender_code':'gender'}, inplace=True)

    elif dataset_name == 'tecent':
        age_dic = {'11~15':0, '16~20':0, '21~25':0, '26~30':1, '31~35':1, '36~40':2, '41~45':2, '46~50':3, '51~55':3, '56~60':4, '61~65':4, '66~70':4, '71~':4}
        df[["age_range"]] = df[["age_range"]].applymap(lambda x:age_dic[x])

        df["age_range"] = df["age_range"].replace(1,0)
        df["age_range"] = df["age_range"].replace(2,1)
        df["age_range"] = df["age_range"].replace(3,1)
        df["age_range"] = df["age_range"].replace(4,1)

        df.rename(columns={'age_level':'age', 'final_gender_code':'gender'}, inplace=True)

    elif dataset_name == 'nba':
        df['SALARY'] = df['SALARY'].replace(-1, 0)
        #df['SALARY'] = df['SALARY'].replace(0, 1)
        #df['SALARY'] = df['SALARY'].replace(1,1)

    # old calculation
    
    total_number_of_sens0 = len(df.loc[df[sens_attr] == 0])
    total_number_of_sens1 = len(df.loc[df[sens_attr] == 1])

    number_of_positive_sens0 = len(df.loc[(df[sens_attr] == 0) & (df[label] == 1)])
    number_of_positive_sens1 = len(df.loc[(df[sens_attr] == 1) & (df[label] == 1)])

    fairness = np.absolute(number_of_positive_sens0) / np.absolute(total_number_of_sens0) - np.absolute(number_of_positive_sens1) / np.absolute(total_number_of_sens1)
    dataset_fainress = fairness * 100
    
    print('dataset fairness:', dataset_fainress)

    
    # new calculation
    #one_df = df[df[sens_attr] == 0]
    #num_of_priv = one_df.shape[0]

    #zero_df = df[df[sens_attr] == 1]
    #num_of_unpriv = zero_df.shape[0]

    #unpriv_outcomes = zero_df[zero_df[label]==1].shape[0]
    #unpriv_ratio = unpriv_outcomes/num_of_unpriv
    

    #priv_outcomes = one_df[one_df[label]==1].shape[0]
    #priv_ratio = priv_outcomes/num_of_priv
    

    #disparate_impact = unpriv_ratio/priv_ratio
    #return disparate_impact
    


    


    pr_unpriv = calc_prop(df, sens_attr, 1, label, 1)
    #print('pr_unpriv: ', pr_unpriv)

    pr_priv = calc_prop(df, sens_attr, 0, label, 1)
    #print('pr_priv:', pr_priv)
    disp = pr_unpriv / pr_priv
    #return pr_unpriv / pr_priv
    print('Dsparate impact:', disp)

    

    #binaryLabelDataset =BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=df, label_names=[label], protected_attribute_names=[sens_attr], unprivileged_protected_attributes=['1'])
    #di = DisparateImpactRemover(repair_level=1.0)
    #rp_train = di.fit_transform(binaryLabelDataset)

    #df_new = rp_train.convert_to_dataframe()[0]



    #print(dataset)
    #print(binaryLabelDataset)
    #return df_new
'''

                                                    


    
