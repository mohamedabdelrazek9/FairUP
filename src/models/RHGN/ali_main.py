from matplotlib.image import imread
import scipy.io
import dgl
import math
import torch
import numpy as np
#from model import *
from RHGN.model import *
import argparse
from sklearn import metrics
import time
from sklearn.metrics import f1_score
import neptune.new as neptune

from RHGN.fairness import Fairness
'''
parser = argparse.ArgumentParser(description='for Alibaba Dataset')
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--n_hid',   type=int, default=32)
parser.add_argument('--n_inp',   type=int, default=200)
parser.add_argument('--clip',    type=int, default=1.0)
parser.add_argument('--max_lr',  type=float, default=1e-2)
parser.add_argument('--label',  type=str, default='gender')
parser.add_argument('--gpu',  type=int, default=0, choices=[0,1,2,3,4,5,6,7])
parser.add_argument('--graph',  type=str, default='G_ori')
parser.add_argument('--model',  type=str, default='RHGN', choices=['RHGN','RGCN'])
parser.add_argument('--data_dir',  type=str, default='../data/sample')
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--sens_attr', type=str, default='gender')
parser.add_argument('--log_tags', type=str, default='')
parser.add_argument('--neptune-project', type=str, default='')
parser.add_argument('--neptune-token', type=str, default='')
parser.add_argument('--multiclass-pred', type=bool, default=False)
parser.add_argument('--multiclass-sens', type=bool, default=False)
args = parser.parse_args()
'''


'''
# Instantiate Neptune client and log arguments
neptune_run = neptune.init(
    project=neptune_project,
    api_token=neptune_token,
)
neptune_run["sys/tags"].add(args.log_tags.split(","))
neptune_run["seed"] = args.seed
neptune_run["dataset"] = "Alibaba-small"
neptune_run["model"] = args.model
neptune_run["label"] = args.label
neptune_run["num_epochs"] = args.n_epoch
neptune_run["n_hid"] = args.n_hid
neptune_run["lr"] = args.max_lr
neptune_run["clip"] = args.clip
'''

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def Batch_train(model, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader, epochs, label, clip, neptune_run):
    tic = time.perf_counter() # start counting time

    best_val_acc = 0
    best_test_acc = 0
    train_step = 0
    Minloss_val = 10000.0
    for epoch in np.arange(epochs) + 1:
        model.train()
        '''---------------------------train------------------------'''
        total_loss = 0
        total_acc = 0
        count = 0
        for input_nodes, output_nodes, blocks in train_dataloader:
            Batch_logits,Batch_labels = model(input_nodes,output_nodes,blocks, out_key='user',label_key=label, is_train=True)

            # The loss is computed only for labeled nodes.
            loss = F.cross_entropy(Batch_logits, Batch_labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_step += 1
            scheduler.step(train_step)

            acc = torch.sum(Batch_logits.argmax(1) == Batch_labels).item()
            total_loss += loss.item() * len(output_nodes['user'].cpu())
            total_acc += acc
            count += len(output_nodes['user'].cpu())

        train_loss, train_acc = total_loss / count, total_acc / count

        if epoch % 1 == 0:
            model.eval()
            '''-------------------------val-----------------------'''
            with torch.no_grad():
                total_loss = 0
                total_acc = 0
                count = 0
                preds=[]
                labels=[]
                for input_nodes, output_nodes, blocks in val_dataloader:
                    Batch_logits,Batch_labels = model(input_nodes, output_nodes,blocks, out_key='user',label_key=label, is_train=False)
                    loss = F.cross_entropy(Batch_logits, Batch_labels)
                    acc   = torch.sum(Batch_logits.argmax(1)==Batch_labels).item()
                    preds.extend(Batch_logits.argmax(1).tolist())
                    labels.extend(Batch_labels.tolist())
                    total_loss += loss.item() * len(output_nodes['user'].cpu())
                    total_acc +=acc
                    count += len(output_nodes['user'].cpu())

                val_f1 = metrics.f1_score(preds, labels, average='macro')
                val_loss,val_acc   = total_loss / count, total_acc / count
                '''------------------------test----------------------'''
                total_loss = 0
                total_acc = 0
                count = 0
                preds = []
                labels = []

                for input_nodes, output_nodes, blocks in test_dataloader:
                    Batch_logits, Batch_labels = model(input_nodes, output_nodes, blocks, out_key='user', label_key=label, is_train=False)
                    loss = F.cross_entropy(Batch_logits, Batch_labels)
                    acc   = torch.sum(Batch_logits.argmax(1)==Batch_labels).item()
                    preds.extend(Batch_logits.argmax(1).tolist())
                    labels.extend(Batch_labels.tolist())
                    total_loss += loss.item() * len(output_nodes['user'].cpu())
                    total_acc +=acc
                    count += len(output_nodes['user'].cpu())
                   

                test_f1 = metrics.f1_score(preds,labels, average='macro')
                test_loss,test_acc   = total_loss / count, total_acc / count
                if  val_acc   > best_val_acc:
                    Minloss_val = val_loss
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                print('Epoch: %d LR: %.5f Loss %.4f, val loss %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
                    epoch,
                    optimizer.param_groups[0]['lr'],
                    train_loss,
                    val_loss,
                    val_acc,
                    best_val_acc,
                    test_acc,
                    best_test_acc,
                ))
                print('\t\tval_f1 %.4f test_f1 \033[1;33m %.4f \033[0m' % (val_f1, test_f1))
            torch.cuda.empty_cache()

    classification_report = metrics.classification_report(labels, preds, digits=4)
    print(classification_report+'end')
    # Classification reports
    confusion_matrix = metrics.confusion_matrix(labels, preds)
    print(confusion_matrix)
    f1 = metrics.f1_score(labels, preds, average='macro')
    print('F1 score:', f1)
    # fpr, tpr, _ = metrics.roc_curve(labels, preds)
    # auc = metrics.auc(fpr, tpr)
    # print("AUC:", auc)

    toc = time.perf_counter() # stop counting time
    elapsed_time = (toc-tic)/60
    #print("\nElapsed time: {:.4f} minutes".format(elapsed_time))

    
    # Log result on Neptune
    neptune_run["test/accuracy"] = best_test_acc
    neptune_run["test/f1_score"] = test_f1
    # neptune_run["test/auc"] = auc
    # neptune_run["test/tpr"] = tpr
    # neptune_run["test/fpr"] = fpr
    neptune_run["conf_matrix"] = confusion_matrix
    neptune_run["elaps_time"] = elapsed_time   

    #fair_obj = Fairness(G, test_idx, targets, predictions, sens_attr, neptune_run, multiclass_pred, multiclass_sens) 
    
    return labels, preds


######################################################################
def ali_training_main(G, cid1_feature, cid2_feature, cid3_feature, model_type, seed, gpu, label, n_inp, batch_size, num_hidden, epochs, lr, sens_attr, multiclass_pred, multiclass_sens, clip, neptune_project, neptune_token):

# Instantiate Neptune client and log arguments
    neptune_run = neptune.init(
        project=neptune_project,
        api_token=neptune_token,
    )
    #neptune_run["sys/tags"].add(args.log_tags.split(","))
    neptune_run["seed"] = seed
    neptune_run["dataset"] = "Alibaba-small"
    #neptune_run["model"] = model
    neptune_run["label"] = label
    neptune_run["num_epochs"] = epochs
    neptune_run["n_hid"] = num_hidden
    neptune_run["lr"] = lr
    neptune_run["clip"] = clip


    '''Fixed random seeds'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    '''Loading charts and labels'''
    #G=torch.load('{}/{}.pkl'.format(args.data_dir,args.graph))
    print(G)
    labels=G.nodes['user'].data[label]
    print(labels.max().item()+1)

    # generate train/val/test split
    pid = np.arange(len(labels))
    shuffle = np.random.permutation(pid)
    train_idx = torch.tensor(shuffle[0:int(len(labels)*0.75)]).long()
    val_idx = torch.tensor(shuffle[int(len(labels)*0.75):int(len(labels)*0.875)]).long()
    test_idx = torch.tensor(shuffle[int(len(labels)*0.875):]).long()

    print("train_idx:", train_idx.shape)
    print("val_idx:", val_idx.shape)
    print("test_idx:", test_idx.shape, type(test_idx), test_idx)

    node_dict = {}
    edge_dict = {}
    for ntype in G.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in G.etypes:
        edge_dict[etype] = len(edge_dict)
        G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    # Initialize input feature
    # import fasttext
    # model = fasttext.load_model('../jd_data/fasttext/fastText/cc.zh.200.bin')
    # sentence_dic=torch.load('../jd_data/sentence_dic.pkl')
    # sentence_vec = [model.get_sentence_vector(sentence_dic[k]) for k, v in enumerate(G.nodes('item').tolist())]
    # for ntype in G.ntypes:
    #     if ntype=='item':
    #         emb=nn.Parameter(torch.Tensor(sentence_vec), requires_grad = False)
    #     else:
    #         emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), 200), requires_grad = False)
    #         nn.init.xavier_uniform_(emb)
    #     G.nodes[ntype].data['inp'] = emb

    for ntype in G.ntypes:
        emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), n_inp), requires_grad = False)
        nn.init.xavier_uniform_(emb)
        G.nodes[ntype].data['inp'] = emb


    G = G.to(device)
    train_idx_item=torch.tensor(shuffle[0:int(G.number_of_nodes('item') * 0.75)]).long()
    val_idx_item = torch.tensor(shuffle[int(G.number_of_nodes('item')*0.75):int(G.number_of_nodes('item')*0.875)]).long()
    test_idx_item = torch.tensor(shuffle[int(G.number_of_nodes('item')*0.875):]).long()
    '''Sampling'''
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    train_dataloader = dgl.dataloading.NodeDataLoader(
        G, {'user':train_idx.to(device)}, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        device=device)

    val_dataloader = dgl.dataloading.NodeDataLoader(
        G, {'user':val_idx.to(device)}, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        device=device)

    test_dataloader = dgl.dataloading.NodeDataLoader(
        G, {'user':test_idx.to(device)}, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        device=device)


    if model_type=='RHGN':
        #cid1_feature = torch.load('{}/cid1_feature.npy'.format(args.data_dir))
        #cid2_feature = torch.load('{}/cid2_feature.npy'.format(args.data_dir))
        #cid3_feature = torch.load('{}/cid3_feature.npy'.format(args.data_dir))

        model = ali_RHGN(G,
                    node_dict, edge_dict,
                    n_inp=n_inp,
                    n_hid=num_hidden,
                    n_out=labels.max().item()+1,
                    n_layers=2,
                    n_heads=4,
                    cid1_feature=cid1_feature,
                    cid2_feature=cid2_feature,
                    cid3_feature=cid3_feature,
                
                    use_norm = True).to(device)
        optimizer = torch.optim.AdamW(model.parameters())

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=epochs,
                                                        steps_per_epoch=int(train_idx.shape[0]/batch_size)+1,max_lr = lr)
        print('Training RHGN with #param: %d' % (get_n_params(model)))
        targets, predictions = Batch_train(model, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader, epochs, label, clip, neptune_run)

        # Compute fairness
        fair_obj = Fairness(G, test_idx, targets, predictions, sens_attr, neptune_run, multiclass_pred, multiclass_sens)
        fair_obj.statistical_parity()
        fair_obj.equal_opportunity()
        fair_obj.overall_accuracy_equality()
        fair_obj.treatment_equality()