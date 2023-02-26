from turtle import forward
import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from FairGNN.src.models.GCN import GCN
from RHGN.layers import *
from RHGN.layers import RHGNLayer

class RHGN_adv(nn.Module):
    def __init__(self, G, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, cid1_feature, cid2_feature, cid3_feature):
        super(RHGN_adv, self).__init__()
        self.cid1_feature = nn.Embedding(cid1_feature.size(0), cid1_feature.size(1))
        self.cid1_feature.weight = nn.Parameter(cid1_feature)
        self.cid1_feature.weight.requires_grad = False

        self.cid2_feature = nn.Embedding(cid2_feature.size(0), cid2_feature.size(1))
        self.cid2_feature.weight = nn.Parameter(cid2_feature)
        self.cid2_feature.weight.requires_grad = False

        self.cid3_feature= nn.Embedding(cid3_feature.size(0), cid3_feature.size(1))
        self.cid3_feature.weight = nn.Parameter(cid3_feature)
        self.cid3_feature.weight.requires_grad = False

        self.adv_model = nn.Linear(n_hid, 1) # was n_out
        #self.sens_model = nn.Linear(64, 2)
        self.sens_model = GCN(200, 128, 1, 0.5)
        #self.optimizer_A = torch.optim.Adam(self.adv_model.parameters(), lr=0.1, weight_decay=1e-5)
        #self.A_loss = 0


    def forward(self, h, inputs, G, blocks, out_key, label_key, is_train=True, print_flag=False):
        # h from orignal model
        #s = self.sens_model(h)
        inputs_new = inputs[0]
        print('graph:', G)
        s = self.sens_model(G, inputs_new)
        print('inputs:', inputs.shape)
        s_g = self.adv_model(h)
        print('s:', s.shape)
        print('s_g:', s_g.shape)
        return s, s_g

class ali_RHGN(nn.Module):
    def __init__(self, G, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads,cid1_feature,cid2_feature,cid3_feature, use_norm = True):
        super(ali_RHGN, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws  = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp,   n_hid))
        for _ in range(n_layers):
            self.gcs.append(RHGNLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm = use_norm))
        self.out = nn.Linear(n_hid, n_out)

        self.cid1_feature= nn.Embedding(cid1_feature.size(0), cid1_feature.size(1))
        self.cid1_feature.weight = nn.Parameter(cid1_feature)
        self.cid1_feature.weight.requires_grad = False

        self.cid2_feature= nn.Embedding(cid2_feature.size(0), cid2_feature.size(1))
        self.cid2_feature.weight = nn.Parameter(cid2_feature)
        self.cid2_feature.weight.requires_grad = False

        self.cid3_feature= nn.Embedding(cid3_feature.size(0), cid3_feature.size(1))
        self.cid3_feature.weight = nn.Parameter(cid3_feature)
        self.cid3_feature.weight.requires_grad = False

  
        self.excitation = nn.Sequential(
            nn.Linear(3, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 3, bias=False),
            nn.ReLU()
        )
        self.query = nn.Linear(200, n_inp)
        self.key = nn.Linear(200, n_inp)
        self.value = nn.Linear(200, n_inp)
        self.skip = nn.Parameter(torch.ones(1))
        print('n_out:', self.n_out)

        #self.query_sens = nn.Linear(200, n_inp)
        #self.key_sens = nn.Linear(200, n_inp)
        #self.value_sens = nn.Linear(200, n_inp)

        #self.adv_model = nn.Linear(128, 1)
        #self.adv_model = nn.Linear(n_hid, n_out)
        #self.sens_model = GCN(95, 128, 1, 0.5)
        #self.sens_model = nn.Linear(n_hid, n_out)
        #self.sens_model2 = nn.Linear(n_inp, n_hid)
        #self.sens_model3 = nn.Linear(n_hid, n_out)

        #self.optimizer_A = torch.optim.Adam(self.adv_model.parameters(), lr=0.1, weight_decay=1e-5)
        #self.criterion = nn.BCEWithLogitsLoss()

        #self.optimizer_G = torch.optim.Adam(self.parameters())

        #self.A_loss = 0
        #self.G_loss = 0

        #self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer_G, epochs=epochs,
        #                                                steps_per_epoch=int(train_idx.shape[0]/batch_size)+1,max_lr = lr)

    def forward(self, input_nodes, output_nodes,blocks, out_key,label_key, is_train=True,print_flag=False):

        item_cid1=blocks[0].srcnodes['item'].data['cid1'].unsqueeze(1)        #(N,1)
        cid1_feature = self.cid1_feature(item_cid1)     #       #(N,1,200)
	

        item_cid2=blocks[0].srcnodes['item'].data['cid2'].unsqueeze(1)        #(N,1)
        cid2_feature = self.cid2_feature(item_cid2)     #       #(N,1,200)

        item_cid3=blocks[0].srcnodes['item'].data['cid3'].unsqueeze(1)        #(N,1)
        cid3_feature = self.cid3_feature(item_cid3)     #       #(N,1,200)
 
        
        cid2_feature=cid1_feature
        cid3_feature=cid1_feature
         
        item_feature = blocks[0].srcnodes['item'].data['inp']
        user_feature = blocks[0].srcnodes['user'].data['inp']
        # brand_feature = blocks[0].srcnodes['brand'].data['inp']

        inputs=torch.cat((cid1_feature,cid2_feature,cid3_feature),1)        #(N,4,200)
        #print('inputs:', inputs.shape) # (455, 3, 200)
        k = self.key(inputs) #(N,4,n_inp)
        v = self.value(inputs) #(N,4,n_inp)
        q = self.query(item_feature.unsqueeze(-2)) #(N,1,n_inp)

        att_score = torch.einsum("bij,bjk->bik", k, q.transpose(1,2)) / math.sqrt(200) #(N,4,1)
        att_score = torch.softmax(att_score, axis=1) # (N,4,1)

        
        alpha = torch.sigmoid(self.skip)    #(1,)
        temp = v * att_score        #(N,4,n_inp)
        item_feature = alpha*(torch.mean(temp, dim=-2).squeeze(-2))  + (1-alpha)*item_feature   # #(N,200)
        #print('item_feature:', item_feature)
        h = {}
        h['item']=F.gelu(self.adapt_ws[self.node_dict['item']](item_feature))
        h['user']=F.gelu(self.adapt_ws[self.node_dict['user']](user_feature))
        # h['brand']=F.gelu(self.adapt_ws[self.node_dict['brand']](brand_feature))

        for i in range(self.n_layers):
            h = self.gcs[i](blocks[i], h, is_train=is_train,print_flag=print_flag)

        h = h[out_key]
        #print('h:', h)
        #self.adv_model.requires_grad_(False)
        #add sens model input
        #s = self.sens_model(inputs)
        #s = self.sens_model2(s)
        #s = self.sens_model3(s)
        #add adv model input
        #s_g = self.adv_model(h)

        h_new=self.out(h)
        #print('h_new:', h_new.shape)
        labels=blocks[-1].dstnodes[out_key].data[label_key]

        # h=F.log_softmax(h, dim=1)
        # return will be h, labels, and estimator output
        return h_new, labels

        


class jd_RHGN(nn.Module):
    def __init__(self, G, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, cid1_feature, cid2_feature,
                 cid3_feature, cid4_feature, use_norm=True, ):
        super(jd_RHGN, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp, n_hid))
        for _ in range(n_layers):
            self.gcs.append(RHGNLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm=use_norm))
        
        self.out = nn.Linear(n_hid, n_out)

        self.cid1_feature = nn.Embedding(cid1_feature.size(0), cid1_feature.size(1))
        self.cid1_feature.weight = nn.Parameter(cid1_feature)
        self.cid1_feature.weight.requires_grad = False

        self.cid2_feature = nn.Embedding(cid2_feature.size(0), cid2_feature.size(1))
        self.cid2_feature.weight = nn.Parameter(cid2_feature)
        self.cid2_feature.weight.requires_grad = False

        self.cid3_feature = nn.Embedding(cid3_feature.size(0), cid3_feature.size(1))
        self.cid3_feature.weight = nn.Parameter(cid3_feature)
        self.cid3_feature.weight.requires_grad = False

        self.cid4_feature = nn.Embedding(cid4_feature.size(0), cid4_feature.size(1))
        self.cid4_feature.weight = nn.Parameter(cid4_feature)
        self.cid4_feature.weight.requires_grad = False

        self.excitation = nn.Sequential(
            nn.Linear(4, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 4, bias=False),
            nn.ReLU()
        )
        self.query = nn.Linear(200, n_inp)
        self.key = nn.Linear(200, n_inp)
        self.value = nn.Linear(200, n_inp)
        self.skip = nn.Parameter(torch.ones(1))
        self.l1=nn.Linear(200, n_inp)
        self.l2=nn.Linear(200, n_inp)
        self.l3=nn.Linear(200, n_inp)
        self.l4=nn.Linear(200, n_inp)

    def forward(self, input_nodes, output_nodes, blocks, out_key, label_key, is_train=True,print_flag=False):

        item_cid1 = blocks[0].srcnodes['item'].data['cid1'].unsqueeze(1)  # (N,1)
        cid1_feature = self.cid1_feature(item_cid1)  # #(N,1,200)
        #cid1_feature = self.l1(cid1_feature)

        item_cid2 = blocks[0].srcnodes['item'].data['cid2'].unsqueeze(1)  # (N,1)
        cid2_feature = self.cid2_feature(item_cid2)  # #(N,1,200)
        #cid2_feature = self.l2(cid2_feature)

        item_cid3 = blocks[0].srcnodes['item'].data['cid3'].unsqueeze(1)  # (N,1)
        cid3_feature = self.cid3_feature(item_cid3)  # #(N,1,200)
        #cid3_fature = self.l3(cid3_feature)

        # item_cid4 = blocks[0].srcnodes['item'].data['brand'].unsqueeze(1)  # (N,1)
        # cid4_feature = self.cid4_feature(item_cid4)  # #(N,1,200)
        #cid4_feature = self.l4(cid4_feature)

        cid2_feature=cid1_feature
        cid3_feature=cid1_feature
        # cid4_feature=cid1_feature

        item_feature = blocks[0].srcnodes['item'].data['inp']
        user_feature = blocks[0].srcnodes['user'].data['inp']

        # inputs = torch.cat((cid1_feature, cid2_feature, cid3_feature, cid4_feature), 1)  # (N,4,200)
        inputs = torch.cat((cid1_feature, cid2_feature, cid3_feature), 1) # (N,3,200)
        k = self.key(inputs)  # (N,3,200)
        v = self.value(inputs)  # (N,3,200)
        q = self.query(item_feature.unsqueeze(-2))  # (N,1,32)

        att_score = torch.einsum("bij,bjk->bik", k, q.transpose(1, 2)) / math.sqrt(200)  # (N,4,1)
        att_score = torch.softmax(att_score, axis=1)  # (N,4,1)

        #Z = torch.mean(inputs, dim=-1, out=None)  # (N,4)
        #A = self.excitation(Z).unsqueeze(-1)  # (N,4,1)
        #att_score = att_score + A  # (N,4,1)
        alpha = torch.sigmoid(self.skip)  # (1,)
        temp = v * att_score  # (N,4,200)
        item_feature = alpha * (torch.mean(temp, dim=-2).squeeze(-2)) + (1 - alpha) * item_feature  # (N,200)

        h = {}
        h['item'] = F.gelu(self.adapt_ws[self.node_dict['item']](item_feature))
        h['user'] = F.gelu(self.adapt_ws[self.node_dict['user']](user_feature))

        for i in range(self.n_layers):
            h = self.gcs[i](blocks[i], h, is_train=is_train,print_flag=print_flag)

        h = h[out_key]
        h = self.out(h)
        labels = blocks[-1].dstnodes[out_key].data[label_key]

        # h=F.log_softmax(h, dim=1)

        return h, labels

