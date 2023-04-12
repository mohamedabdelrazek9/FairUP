## Main File for pre-processing component
## this file will pre-process the choosen data for either all models or the user-chosen models and begin the training for each choosen model

import argparse
import os
#from turtle import st
from src.utils import bin_alibaba, load_networkx_file, load_neo4j_file
from FairGNN.src.utils import load_pokec, feature_norm
from FairGNN.src.train_fairGNN import train_FairGNN
from alibaba_processing.ali_RHGN_pre_processing import ali_RHGN_pre_process
from alibaba_processing.ali_CatGCN_pre_processing import ali_CatGCN_pre_processing
from tecent_processing.tecent_RHGN_pre_processing import tec_RHGN_pre_process
from tecent_processing.tecent_CatGCN_pre_processing import tec_CatGCN_pre_process
from nba_processing.nba_RHGN_pre_processing import nba_RHGN_pre_process 
from nba_processing.nba_CatGCN_pre_processing import nba_CatGCN_pre_process
from pokec_processing.pokec_RHGN_pre_processing import pokec_z_RHGN_pre_process
from pokec_processing.pokec_CatGCN_pre_processing import pokec_z_CatGCN_pre_process
from RHGN.ali_main import ali_training_main
from RHGN.jd_main import tecent_training_main
from CatGCN.train_main import train_CatGCN
from fainress_component import fairness_calculation, disparate_impact_remover, reweighting, lfr
import dgl
import torch
import pandas as pd
from utils import create_edges, bin_age_range_tecent, apply_bin_columns, apply_cat_columns
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
# Todo add arguments for the pre-processing
parser.add_argument('--type', type=int, default=0, choices=[0, 1, 2], help="choose if you want to run the frameowkr 0 for all models or 1, and 2 models")
#parser.add_argument('--model_type', type=str, choices=['FairGNN', 'CatGCN', 'RHGN'], help="only for the case if 1 or 2 models are choosen then we choose from either FairGNN, CatGCN, RHGN")
parser.add_argument('--model_type', nargs='+', default=[])
parser.add_argument('--dataset_name', type=str, choices=['pokec_z', 'pokec_n', 'nba', 'alibaba', 'tecent'], help="choose which dataset you want to apply on the models")
parser.add_argument('--dataset_path', type=str, help="choose which dataset you want to apply on the models")
parser.add_argument('--dataset_user_id_name', type=str, help="The column name of the user in the orginal dataset (e.g. user_id or userid)")
parser.add_argument('--sens_attr', type=str, help="choose which sensitive attribute you want to consider for the framework")
parser.add_argument('--predict_attr', type=str, help="choose which prediction attribute you want to consider for the framework")
parser.add_argument('--label_number', type=int)
parser.add_argument('--sens_number', type=int)
parser.add_argument('--num-hidden', type=int, default=64, help='Number of hidden units of classifier.')
parser.add_argument('--dropout', type=float, default=.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units of the sensitive attribute estimator')
parser.add_argument('--model', type=str, default="GAT", help='the type of model GCN/GAT') ## specific for FairGNN
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--alpha', type=float, default=4, help='The hyperparameter of alpha')
parser.add_argument('--beta', type=float, default=0.01, help='The hyperparameter of beta')
parser.add_argument('--roc', type=float, default=0.745, help='the selected FairGNN ROC score on val would be at least this high')
parser.add_argument('--epochs_rhgn', type=int, default=2000, help='Number of epochs to train')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
parser.add_argument('--acc', type=float, default=0.688, help='the selected FairGNN accuracy on val would be at least this high')
#parser.add_argument('--apply_onehot', type=bool, required=False, help='Decide weather you want the framework to apply one-hot encoding to the data for FairGNN or not (We recommend that the user does the this step and transform the data to either one of the networkx format or neo4j)')
parser.add_argument('--uneeded_columns', nargs="+", help="(OPTIONAL) choose which columns that will not be needed in the dataset and the fairness experiment (e.g. description)")
parser.add_argument('--onehot_bin_columns', nargs="+", help='(OPTIONAL) Decide which of the columns of your dataset are binary (e.g. False/True) to be later on processed')
parser.add_argument('--onehot_cat_columns', nargs="+", help='(OPTIONAL) choose which columns in the dataset will be transofrmed as one-hot encoded')
parser.add_argument('--calc_fairness', type=bool, default=False)
parser.add_argument('--debaising_approach', type=str, choices=['disparate_impact_remover', 'reweighting', 'sample'], help="choose which debaising approach to use while preprocessing the dataset")
#################
# for RHGN
#n_epoch --> epochs
parser.add_argument('--batch_size', type=int, default=512)
#n_hidden --> num_hidden
parser.add_argument('--n_inp',   type=int, default=200)
parser.add_argument('--clip',    type=int, default=1.0)
#max_lr --> lr
parser.add_argument('--label',  type=str, default='gender')
parser.add_argument('--gpu',  type=int, default=0, choices=[0,1,2,3,4,5,6,7])
parser.add_argument('--graph',  type=str, default='G_ori')
# model ---> model_type
#data_dir --> dataset_path
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--log_tags', type=str, default='')
parser.add_argument('--multiclass-pred', type=bool, default=False)
parser.add_argument('--multiclass-sens', type=bool, default=False)
#####################

# for CatGCN
parser.add_argument('--diag-probe', type = float,default = 1., help = "Diag probe coefficient. Default is 1.0.")
parser.add_argument('--graph-refining', nargs = "?", default='agc', help="Optimize the field feature, use 'agc', 'fignn', or 'none'.")
parser.add_argument('--aggr-pooling', nargs = "?", default='mean', help="Aggregate the field feature. Default is 'mean'.")
parser.add_argument("--grn-units",type=str, default="64", help="Hidden units for global interaction modeling, splitted with comma, maybe none.")
parser.add_argument('--bi-interaction', nargs = "?",default='nfm', help="Compute the user feature with nfm, use 'nfm' or 'none'.")
parser.add_argument("--nfm-units",type=str, default="64", help="Hidden units for local interaction modeling, splitted with comma, maybe none.")
parser.add_argument('--graph-layer', nargs = "?",default='sgc', help="Optimize the user feature, use 'pna', 'sgc', 'appnp', etc.")
parser.add_argument("--gnn-hops", type = int, default = 1, help = "Hops number of pure neighborhood aggregation. Default is 1.")
parser.add_argument("--gnn-units",type=str, default="64", help="Hidden units for baseline models, splitted with comma, maybe none.")
parser.add_argument('--aggr-style', nargs = "?", default='sum', help="Aggregate the user feature, use 'sum' or 'none'.")
parser.add_argument("--balance-ratio", type = float, default = 0.5, help = "Balance ratio parameter when aggr_style is 'sum'. Default is 0.5.")
parser.add_argument('--weight-balanced', nargs = "?", default='True', help="Adjust weights inversely proportional to class frequencies.")
parser.add_argument("--clustering-method", nargs = "?", default = "none", help = "Clustering method for graph decomposition, use 'metis', 'random', or 'none'.")
parser.add_argument("--train-ratio", type = float, default = 0.8, help = "Train data ratio. Default is 0.8.")
#parser.add_argument("--patience", type = int, default = 10, help = "Number of training patience. Default is 10.")
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument("--cluster-number", type = int, default = 100, help = "Number of clusters extracted. Default is 100.")
parser.add_argument("--field-dim", type = int, default = 64, help = "Number of field dims. Default is 64.")
parser.add_argument("--num-steps", type = int, default = 2, help = "GRU steps for FiGNN. Default is 2.")
parser.add_argument("--multi-heads", type=str, default="8,1", help="Multi heads in each gat layer, splitted with comma.")
parser.add_argument("--theta", type = float, default = 0.5,  help = "Theta coefficient for GCNII. Default is 0.5.")
parser.add_argument("--gat-units", type=str, default="64", help="Hidden units for global gat part, splitted with comma, maybe none.")

parser.add_argument("--special_case", type=bool, default=False)

parser.add_argument("--num-heads", type=int, default=1,
                        help="number of hidden attention heads")
parser.add_argument("--num-out-heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num-layers", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--residual", action="store_true", default=False,
                    help="use residual connection")
parser.add_argument("--attn-drop", type=float, default=.0,
                    help="attention dropout")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")


parser.add_argument('--neptune_project', type=str, default='')
parser.add_argument('--neptune_token', type=str, default='')
parser.add_argument('--multiclass_pred', type=bool, default=False)
parser.add_argument('--multiclass_sens', type=bool, default=False)


import networkx as nx
import numpy as np

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()

networkx_format_list = ['.graphml', '.gexf', '.gml', '.leda', '.net']
data_extension = os.path.splitext(args.dataset_path)[1]

def FairGNN_pre_processing(data_extension):
    # todo do suitable pre-processing for the choosen dataset
    # check if data is in form of networkx (.graphml) or neo4j
    # Train FairGNN model
    model_type = args.model_type[args.model_type.index('FairGNN')]
    print('Loading dataset for FairGNN...')
    
    # calculate fairness before doing anything in the dataset
    #predict_attr = args.predict_attr
    ###################################################
    #!! Fairness calculation before pre-processing
    ###################################################
    #if(args.calc_fairness):
        #fairness_calculation(args.dataset_name, args.dataset_path, args.sens_attr, predict_attr)

    if data_extension in networkx_format_list:
        df_nodes, edges_path = load_networkx_file(model_type,
                                                  data_extension, 
                                                  args.dataset_name,
                                                  args.dataset_path,
                                                  args.dataset_user_id_name, 
                                                  args.onehot_bin_columns, 
                                                  args.onehot_cat_columns,
                                                  args.sens_attr,
                                                  args.predict_attr)
        # this here needs to be moved after the else condition
        adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_pokec(df_nodes,
                                                                                            edges_path,
                                                                                            args.dataset_user_id_name, 
                                                                                            args.sens_attr, 
                                                                                            args.predict_attr, 
                                                                                            args.label_number, 
                                                                                            args.sens_number,
                                                                                            args.seed,
                                                                                            test_idx=True)
    elif data_extension == '.json':
        df_nodes, edges_path = load_neo4j_file(model_type,
                                               args.dataset_path, 
                                               args.dataset_name,
                                               args.uneeded_columns, 
                                               args.onehot_bin_columns, 
                                               args.onehot_cat_columns)   

    else: # special case we read the original data
        if args.special_case == True:
            print('we will read normal data')
            df_nodes = pd.read_csv(args.dataset_path)
            print('Dataset is read')
            if args.dataset_name == 'tecent':
                df_nodes = bin_age_range_tecent(df_nodes)
                df_nodes = df_nodes.drop(columns=["cid1_name", "cid2_name", "cid3_name", "item_name", "seg_name"])
                edges_path = create_edges(df_nodes, args.dataset_name)
                df_edge_list = edges_path
            elif args.dataset_name == 'nba':
                if args.onehot_bin_columns is not None:
                    df_nodes = apply_bin_columns(df_nodes, args.onehot_bin_columns)
                if args.onehot_cat_columns is not None:
                    df_nodes = apply_cat_columns(df_nodes, args.onehot_cat_columns)
                df_edge_list = pd.read_csv('../nba_relationship.txt', sep=" ", header=None)
                edges_path = '../nba_relationship'
            elif args.dataset_name == 'alibaba':
                #sample 
                #df_nodes = df_nodes.sample(frac=0.10, random_state=11)
                print(df_nodes.shape)
                df_nodes = bin_alibaba(df_nodes)
                edges_path = create_edges(df_nodes, args.dataset_name)
                df_edge_list = edges_path
            elif args.dataset_name == 'pokec_z':
                df_nodes = pd.read_csv(args.dataset_path)
                edges_path = '../region_job_relationship'
                #df_edge_list = edges_path
                #df_edge_list = pd.read_csv('../Master-Thesis-dev/region_job_relationship.txt')
                #df_edge_list.to_csv(r'{}.txt'.format(edges_path), header=None, index=None, sep=' ', mode='a')
                #df = pd.read_csv('../Master-Thesis-dev/region_job.csv')
                #df_edge_list = pd.read_csv('../region_job_relationship.txt',  sep="\t", header=None)
                #edges_path = ''
            #save the edges as .txt file
            #edges_path = './FairGNN_data_relationship'
            # df_edge_list.to_csv(r'{}.txt'.format(edges_path), header=None, index=None, sep=' ', mode='a')
        else:
            # simple test for pokec/tecent
            df_nodes = pd.read_csv(args.dataset_path)    
            #edges_path = pd.read_csv('../region_job_relationship.txt', delimiter="\t", header=None)
            #edges_path.rename(columns={0: "source", 1: "target"}, inplace=True)  
            #edges_path = '../user_edges.csv'
            edges_path = '../region_job_relationship'

        

        adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_pokec(df_nodes,
                                                                                            edges_path,
                                                                                            args.dataset_user_id_name, 
                                                                                            args.sens_attr, 
                                                                                            args.predict_attr, 
                                                                                            args.dataset_name,
                                                                                            args.label_number, 
                                                                                            args.sens_number,
                                                                                            args.seed,
                                                                                            test_idx=True)

    G = dgl.DGLGraph()
    #G.from_scipy_sparse_matrix(adj) # not supported
    G = dgl.from_scipy(adj)
    
    #if args.dataset_name == 'nba' and args.dataset_name == 'alibaba':
    #    features = feature_norm(features)

    if args.dataset_name == 'nba':
        features = feature_norm(features)

    if args.dataset_name == 'nba' or args.dataset_name == 'pokec_z' or args.dataset_name == 'pokec_n':
        labels[labels>1]=1
        if args.sens_attr:
            sens[sens>0]=1

    print('Starting FairGNN training')
    # define Model and optimizer and train
    train_FairGNN(G, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train, args.dataset_name, args.sens_number, args)

    return print('Training FairGNN is done')


def CatGCN_pre_processing(data_extension):

    model_type = args.model_type[args.model_type.index('CatGCN')]

    # todo do suitable pre-processing for the choosen dataset
    print('Loading dataset for CatGCN...')

    predict_attr = args.label
    #fairness_calculation(args.dataset_name, args.dataset_path, args.sens_attr, predict_attr)


    if data_extension in networkx_format_list:
        df, df_edge_list = load_networkx_file(model_type, 
                                data_extension,
                                args.dataset_name,
                                args.dataset_path,  
                                args.dataset_user_id_name,
                                onehot_bin_columns=None,
                                onehot_cat_columns=None)

    elif data_extension == '.json':
        df, df_edge_list = load_neo4j_file(model_type, 
                             args.dataset_path, 
                            args.dataset_name,
                            args.dataset_user_id_name,
                            onehot_bin_columns=None,
                            onehot_cat_columns=None)
    else:
        if args.special_case == True:
            print('we will read normal data')
            if args.dataset_name == 'tecent':
                df_user = pd.read_csv('../user')
                df_click = pd.read_csv('../user_click')
                df_item = pd.read_csv('../item_info')
                df = ''
            elif args.dataset_name == 'alibaba':
                df_user = pd.read_csv('../Master-Thesis-dev/user_profile.csv', usecols=[0,3,4,5,7,8])
                df_click = pd.read_csv('../Master-Thesis-dev/raw_sample.csv', usecols=['user', 'adgroup_id', 'clk'])
                df_item = pd.read_csv('../Master-Thesis-dev/ad_feature.csv', usecols=['adgroup_id', 'cate_id'])
                df = ''
            elif args.dataset_name == 'nba':
                df = pd.read_csv('../nba.csv')
                df_edge_list = pd.read_csv('../nba_relationship.txt', sep="\t", header=None)
                df_edge_list = df_edge_list.rename(columns={0: "source", 1: "target"})
            elif args.dataset_name == 'pokec_z':
                df = pd.read_csv('../Master-Thesis-dev/region_job.csv')
                df_edge_list = pd.read_csv('../new_edges_pokec_catgcn2.txt',  sep="\t")
                df_edge_list = df_edge_list.drop(['Unnamed: 0'], axis=1)
                
                #df_edge_list = pd.read_csv('../region_job_relationship.txt',  sep="\t", header=None)
               
        else:
            #simple test for pokec
            df = pd.read_csv(args.dataset_path)
            #df_edge_list = pd.read_csv('./region_job_relationship.txt', sep=" ", header=None)
            #df_edge_list = pd.read_csv('./region_job_relationship.txt', delimiter= "\t", header=None)
            df_edge_list = pd.read_csv('../region_job_relationship.txt', delimiter= "\t", header=None)
            df_edge_list.rename(columns={0: "source", 1: "target"}, inplace=True)
            #df_edge_list = pd.read_csv('../user_edge.csv')

         
    
    if args.dataset_name == 'alibaba':
        user_edge_path, user_field_path, user_gender_path, user_labels_path = ali_CatGCN_pre_processing(df, df_user, df_click, df_item, args.sens_attr, args.label, args.special_case, args.debaising_approach)
        target = user_gender_path
    elif args.dataset_name == 'tecent':
        user_edge_path, user_field_path, user_gender_path, user_labels_path = tec_CatGCN_pre_process(df, df_user, df_click, df_item, args.sens_attr, args.label, args.special_case, args.debaising_approach)
        target = user_gender_path

    # Todo implment CatGCN processing for NBA dataset
    elif args.dataset_name == 'nba':
        user_edge_path, user_field_path, user_salary_path, user_labels_path = nba_CatGCN_pre_process(df, df_edge_list, args.sens_attr, args.label, args.special_case, args.onehot_bin_columns , args.onehot_cat_columns,args.debaising_approach)
        target = user_salary_path

    # Todo implment CatGCN processing for Pokec dataset
    elif args.dataset_name == 'pokec_z':
        user_edge_path, user_field_path, user_work_path, user_labels_path = pokec_z_CatGCN_pre_process(df, df_edge_list, args.sens_attr, args.label, args.debaising_approach)
        target = user_work_path

    #print('Dataset fairness before training:', dataset_fairness)
    
    # Add model training after data processing
    print('Starting CatGCN training')
    print('show neptune:', args.neptune_project)
    train_CatGCN(user_edge_path, user_field_path, target, user_labels_path, args.seed, args.label, args)
    
    return print('Training CatGCN is done.')


def RHGN_pre_processing(data_extension):
    # todo do suitable pre-processing for the choosen dataset
    
    model_type = args.model_type[args.model_type.index('RHGN')]

    print('Loading dataset for RHGN...')

    predict_attr = args.label
    #fairness_calculation(args.dataset_name, args.dataset_path, args.sens_attr, predict_attr)

    if data_extension in networkx_format_list:
        df = load_networkx_file(model_type,
                                data_extension,
                                args.dataset_name,
                                args.dataset_path, 
                                args.dataset_user_id_name,
                                onehot_bin_columns=None,
                                onehot_cat_columns=None) #argument may change
        # todo later on: add condition for other datasets
    elif data_extension == '.json':
        df = load_neo4j_file(model_type, 
                             args.dataset_path, 
                             args.dataset_name)

    else:
        if args.special_case == True:
            print('we will read normal data')
            if args.dataset_name == 'tecent':
                df_user = pd.read_csv('../user')
                df_click = pd.read_csv('../user_click')
                df_item = pd.read_csv('../item_info')
                df = ''
            elif args.dataset_name == 'alibaba':
                df_user = pd.read_csv('../Master-Thesis-dev/user_profile.csv', usecols=[0,3,4,5,7,8])
                df_click = pd.read_csv('../Master-Thesis-dev/raw_sample.csv', usecols=['user', 'adgroup_id', 'clk'])
                df_item = pd.read_csv('../Master-Thesis-dev/ad_feature.csv')
                df_item.dropna(axis=0, subset=['cate_id', 'campaign_id', 'brand'], inplace=True)
                df = ''
            elif args.dataset_name == 'nba':
                df = pd.read_csv('../nba.csv')
                df_edge_list = pd.read_csv('../nba_relationship.txt', sep="\t", header=None)
                df_edge_list = df_edge_list.rename(columns={0: "source", 1: "target"})
            elif args.dataset_name == 'pokec_z':
                df = pd.read_csv('../Master-Thesis-dev/region_job.csv')
                df_edge_list = pd.read_csv('../region_job_relationship.txt',  sep="\t", header=None)
        else:
            #df = pd.read_csv(args.dataset_path)
            #df_user = ''
            #df_click = ''
            #df_item = ''
            df = pd.read_csv(args.dataset_path)
            df_edge_list = pd.read_csv('../region_job_relationship.txt', delimiter= "\t", header=None)
            df_edge_list.rename(columns={0: "source", 1: "target"}, inplace=True)
    #else: # simple test for pokec
    #    df = pd.read_csv(args.dataset_path)
    
    #print('Dataset fairness before training:', dataset_fairness)

    '''
    if args.debaising_approach:
        if args.debaising_approach == 'disparate_impact_remover':
            print('columns:', df.columns.tolist())
            print('')
            df = disparate_impact_remover(df, args.sens_attr, args.label)
        elif args.debaising_approach == 'reweighting':
            df = reweighting(df, args.sens_attr, args.label)
        elif args.debaising_approach == 'lfr':
            df = lfr(df, args.sens_attr, args.label)
    '''
    if args.dataset_name == 'alibaba':
        #G, cid1_feature, cid2_feature, cid3_feature = ali_RHGN_pre_process(df)
        G, cid1_feature, cid2_feature, cid3_feature = ali_RHGN_pre_process(df, df_user, df_click, df_item, args.sens_attr, args.label, args.special_case, args.debaising_approach)
    elif args.dataset_name == 'tecent':
        G, cid1_feature, cid2_feature, cid3_feature, cid4_feature = tec_RHGN_pre_process(df, df_user, df_click, df_item, args.sens_attr, args.label, args.special_case, args.debaising_approach)

    # Todo implment RHGN processing for NBA dataset
    elif args.dataset_name == 'nba':
        G, cid1_feature, cid2_feature, cid3_feature = nba_RHGN_pre_process(df, args.dataset_user_id_name, args.sens_attr, args.label, args.onehot_bin_columns, args.onehot_cat_columns, args.debaising_approach)


    # Todo implment RHGN processing for Pokec dataset
    elif args.dataset_name == 'pokec_z':
        G, cid1_feature, cid2_feature, cid3_feature = pokec_z_RHGN_pre_process(df, args.dataset_user_id_name, args.sens_attr, args.label, args.debaising_approach)


    


    # Add model training after data processing
    print('Starting RHGN training')
    if args.dataset_name == 'tecent':
        tecent_training_main(G,
                            cid1_feature,
                            cid2_feature,
                            cid3_feature,
                            cid4_feature,
                            model_type,
                            args.seed,
                            args.gpu,
                            args.label,
                            args.n_inp,
                            args.batch_size,
                            args.num_hidden,
                            args.epochs_rhgn,
                            args.lr,
                            args.sens_attr,
                            args.multiclass_pred,
                            args.multiclass_sens,
                            args.clip)

    else:
        ali_training_main(G, 
                        cid1_feature, 
                        cid2_feature, 
                        cid3_feature,
                        model_type,
                        args.seed, 
                        args.gpu, 
                        args.label, 
                        args.n_inp, 
                        args.batch_size, 
                        args.num_hidden, 
                        args.epochs_rhgn, 
                        args.lr, 
                        args.sens_attr, 
                        args.multiclass_pred, 
                        args.multiclass_sens, 
                        args.clip,
                        args.neptune_project,
                        args.neptune_token)

    return print('Training RHGN is done.')

# not needed, model can work with only given the names of model
#if args.type == 0:
#    fair_pre_processing = FairGNN_pre_processing(data_extension)
#    cat_pre_processing = CatGCN_pre_processing(data_extension)
#    rhgn_pre_processing = RHGN_pre_processing(data_extension)

if args.type == 1:
    if(args.calc_fairness):
        fairness_calculation(args.dataset_name, args.dataset_path, args.sens_attr, args.predict_attr)
    #if 'FairGNN' in args.model_type:
        fair_pre_processing = FairGNN_pre_processing(data_extension)
    if 'CatGCN' in args.model_type:
        cat_pre_processing = CatGCN_pre_processing(data_extension)
    #if 'RHGN' in args.model_type:
        rhgn_pre_processing = RHGN_pre_processing(data_extension)
    if 'FairGNN' in args.model_type and 'RHGN' in args.model_type:
        fair_pre_processing = FairGNN_pre_processing(data_extension)
        rhgn_pre_processing = RHGN_pre_processing(data_extension)

#if 'FairGNN' in args.model_type:
#    fair_pre_processing = FairGNN_pre_processing(data_extension)
#    cat_pre_processing = CatGCN_pre_processing(data_extension)
#    rhgn_pre_processing = RHGN_pre_processing(data_extension)

# not needed, model can work with only given the names of model
#elif args.type == 2:
#    if args.model_type == 'FairGNN' and args.model_type == 'CatGCN':
#        fair_pre_processing = FairGNN_pre_processing(data_extension)
#        cat_pre_processing = CatGCN_pre_processing(data_extension)

#    if args.model_type == 'FairGNN' and args.model_type == 'RHGN':
#        fair_pre_processing = FairGNN_pre_processing(data_extension)
#        rhgn_pre_processing = RHGN_pre_processing(data_extension)

#    if 'CatGCN' in args.model_type and 'RHGN' in args.model_type:
#        cat_pre_processing = CatGCN_pre_processing(data_extension)
#        rhgn_pre_processing = RHGN_pre_processing(data_extension)
