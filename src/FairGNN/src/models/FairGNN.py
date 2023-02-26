import torch.nn as nn
from FairGNN.src.models.GCN import GCN, GCN_Body
from FairGNN.src.models.GAT import GAT, GAT_body
import torch

def get_model(nfeat, args):
    if args.model == "GCN":
        model = GCN_Body(nfeat,args.num_hidden,args.dropout)
    elif args.model == "GAT":
        heads =  ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT_body(args.num_layers,nfeat,args.num_hidden,heads,args.dropout,args.attn_drop,args.negative_slope,args.residual)
    else:
        print("Model not implement")
        return

    return model

class FairGNN(nn.Module):

    def __init__(self, nfeat, args):
        super(FairGNN,self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        #print('nfeat:', nfeat)
        #print('hidden:', args.hidden)
        self.estimator = GCN(nfeat,args.hidden,1,dropout)
        self.GNN = get_model(nfeat,args)
        self.classifier = nn.Linear(nhid,1)
        self.adv = nn.Linear(nhid,1)

        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters()) + list(self.estimator.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)
        self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr = args.lr, weight_decay = args.weight_decay)

        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()

        self.G_loss = 0
        self.A_loss = 0

    def forward(self,g,x):
        s = self.estimator(g,x)
        z = self.GNN(g,x)
        y = self.classifier(z)
        return y,s
    
    def optimize(self,g,x,labels,idx_train,sens,idx_sens_train):
        # x ==  features
        self.train()

        ### update E, G
        self.adv.requires_grad_(False)
        self.optimizer_G.zero_grad()

        #print('g:', g.shape)
        #print('x:', x.shape)
        #print('G:', g)
        s = self.estimator(g,x)
        h = self.GNN(g,x)
        y = self.classifier(h)



        s_g = self.adv(h)

        s_score = torch.sigmoid(s.detach())
        # s_score = (s_score > 0.5).float()
        #print('s_score:', s_score.shape)
        #print('idx_sens_tran:', idx_sens_train.shape)
        #print('sens:', sens.shape)
        #print('sens[idx_sens_train]:', sens[idx_sens_train])
        #print('shape:', sens[idx_sens_train].shape)
        #print('')
        s_score[idx_sens_train]=sens[idx_sens_train].unsqueeze(1).float()
        #print('s_score[idx_sens_train]:', s_score[idx_sens_train])
        #print('shape:', s_score[idx_sens_train].shape)
        y_score = torch.sigmoid(y)
        # Lr
        self.cov =  torch.abs(torch.mean((s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))))


        self.cls_loss = self.criterion(y[idx_train],labels[idx_train].unsqueeze(1).float())
        #print('')
        #print('s_g shape:', s_g.shape)
        #print('s_score shape:', s_score.shape)
        self.adv_loss = self.criterion(s_g,s_score)                
        
        # calculate G loss
        self.G_loss = self.cls_loss  + self.args.alpha * self.cov - self.args.beta * self.adv_loss
        self.G_loss.backward()
        self.optimizer_G.step()

        ## update Adv
        self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()
        s_g = self.adv(h.detach())

        # the Adversary loss --> the goal is to maxmize the loss of the Adversary -- to let it make false prediction
        self.A_loss = self.criterion(s_g,s_score)
        self.A_loss.backward()
        self.optimizer_A.step()

