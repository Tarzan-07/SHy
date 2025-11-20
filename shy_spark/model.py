import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import numpy as np
from layers import *
class HeirarchEmbedding(nn.Module):
    def __init__(self, code_levels: np.ndarray, max_vals, code_dims):
        super(HeirarchEmbedding, self).__init__()
        self.code_levels = code_levels
        self.max_len = len(max_vals)
        self.hierEmbeddings = nn.ModuleList()
        self.hierEmbeddings.append(nn.Embedding(code_num, code_dim) for level, (code_num, code_dim) in enumerate(zip(max_vals, code_dims)) )

    def forward(self):
        embeddings = [self.hierEmbeddings[level](self.code_levels[:, level] - 1) for level in range(self.max_len)]
        hierairchial_embeddings = torch.cat(embeddings, dim=1)
        
        return hierairchial_embeddings



class HGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, nhead, dropout_p, hgnn_model, device):
        super(HGNN, self).__init__()
        self.nlayer = nlayer
        self.HGNN_model = hgnn_model
        if hgnn_model == 'UniGINConv':
            self.convs = nn.ModuleList(
                [UniGINConv(nfeat, nhid, heads=nhead, dropout=0.)] +
                [UniGINConv(nhid * nhead, nhid, heads=nhead, dropout=0.) for _ in range(self.nlayer - 1)]
            )
            if self.nlayer > 0:
                self.conv_out = UniGINConv(nhid * nhead, nclass, heads=1, dropout=0.)
            else:
                self.conv_out = UniGINConv(nfeat, nclass, heads=1, dropout=0.)
        elif hgnn_model == 'UniSAGEConv':
            self.convs = nn.ModuleList(
                [UniSAGEConv(nfeat, nhid, heads=nhead, dropout=0.)] +
                [UniSAGEConv(nhid * nhead, nhid, heads=nhead, dropout=0.) for _ in range(self.nlayer - 1)]
            )
            if self.nlayer > 0:
                self.conv_out = UniSAGEConv(nhid * nhead, nclass, heads=1, dropout=0.)
            else:
                self.conv_out = UniSAGEConv(nfeat, nclass, heads=1, dropout=0.)
        elif hgnn_model == 'UniGATConv':
            self.convs = nn.ModuleList(
                [UniGATConv(nfeat, nhid, heads=nhead, dropout=0.)] +
                [UniGATConv(nhid * nhead, nhid, heads=nhead, dropout=0.) for _ in range(self.nlayer - 1)]
            )
            if self.nlayer > 0:
                self.conv_out = UniGATConv(nhid * nhead, nclass, heads=1, dropout=0.)
            else:
                self.conv_out = UniGATConv(nfeat, nclass, heads=1, dropout=0.)
        elif hgnn_model == 'UniGCNConv':
            self.convs = nn.ModuleList(
                [UniGCNConv(nfeat, nhid, heads=nhead, dropout=0.)] +
                [UniGCNConv(nhid * nhead, nhid, heads=nhead, dropout=0.) for _ in range(self.nlayer - 1)]
            )
            if self.nlayer > 0:
                self.conv_out = UniGCNConv(nhid * nhead, nclass, heads=1, dropout=0.)
            else:
                self.conv_out = UniGCNConv(nfeat, nclass, heads=1, dropout=0.)
        elif hgnn_model == 'UniGCNIIConv':
            self.prelude = nn.Linear(nfeat, nhid)
            self.convs = nn.ModuleList(
                [UniGCNIIConv(nhid, nhid, heads=nhead, dropout=0.)] +
                [UniGCNIIConv(nhid, nhid, heads=nhead, dropout=0.) for _ in range(self.nlayer - 1)]
            )
            if self.nlayer > 0:
                self.conv_out = UniGCNIIConv(nhid, nhid, heads=1, dropout=0.)
                self.postlude = nn.Linear(nhid, nclass)
            else:
                self.conv_out = UniGCNIIConv(nfeat, nfeat, heads=1, dropout=0.)
                self.postlude = nn.Linear(nfeat, nclass)
        elif hgnn_model == 'AllDeepSets':
            self.convs = nn.ModuleList(
                [AllSet(nfeat, nhid, heads=nhead, aggr='add', PMA=False, device=device, dropout=dropout_p)] +
                [AllSet(nhid, nhid, heads=nhead, aggr='add', PMA=False, device=device, dropout=dropout_p) for _ in range(self.nlayer - 1)]
            )
            if self.nlayer > 0:
                self.conv_out = AllSet(nhid, nclass, heads=nhead, aggr='add', PMA=False, device=device, dropout=dropout_p)
            else:
                self.conv_out = AllSet(nfeat, nclass, heads=nhead, aggr='add', PMA=False, device=device, dropout=dropout_p)
        elif hgnn_model == 'AllSetTransformer':
            self.convs = nn.ModuleList(
                [AllSet(nfeat, nhid, heads=nhead, aggr='mean', PMA=True, device=device, dropout=dropout_p)] +
                [AllSet(nhid, nhid, heads=nhead, aggr='mean', PMA=True, device=device, dropout=dropout_p) for _ in range(self.nlayer - 1)]
            )
            if self.nlayer > 0:
                self.conv_out = AllSet(nhid, nclass, heads=nhead, aggr='mean', PMA=True, device=device, dropout=dropout_p)
            else:
                self.conv_out = AllSet(nfeat, nclass, heads=nhead, aggr='mean', PMA=True, device=device, dropout=dropout_p)
        elif hgnn_model == 'HyperGCNConv':
            self.convs = nn.ModuleList(
                [HyperGCNConv(nfeat, nhid, True, device, dropout_p)] +
                [HyperGCNConv(nfeat, nhid, True, device, dropout_p) for _ in range(self.nlayer - 1)]
            )
            if self.nlayer > 0:
                self.conv_out = HyperGCNConv(nhid, nclass, True, device, dropout_p)
            else:
                self.conv_out = HyperGCNConv(nfeat, nclass, True, device, dropout_p)
        else:
            print("Error: no selected hypergraph neural network model.")
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, X, V, E, H):
        if self.HGNN_model == "UniGCNConv":
            if self.nlayer > 0:
                for conv in self.convs:
                    X = conv(X, V, E, H)
                    X = self.act(X)
                    X = self.dropout(X)
            X = self.conv_out(X, V, E, H)
        elif self.HGNN_model == "UniGCNIIConv":
            if self.nlayer > 0:
                X = F.relu(self.prelude(X))
                X0 = X
                for conv in self.convs:
                    X = conv(X, V, E, X0, H)
                    X = self.act(X)
                    X = self.dropout(X)
                X = self.conv_out(X, V, E, X0, H)
                X = self.postlude(X)
            else:
                X = self.conv_out(X, V, E, X, H)
                X = self.postlude(X)
        else:
            if self.nlayer > 0:
                for conv in self.convs:
                    X = conv(X, V, E)
                    X = self.act(X)
                    X = self.dropout(X)
            X = self.conv_out(X, V, E)
        return F.leaky_relu(X)


class hslencoder():
    def __init__(self):
        pass

    def forward():
        return
    
class hsldecoder():
    def __init__(self):
        pass

    def forward():
        return

class FC(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward():
        return

class shy(nn.Module):
    def __init__(self, code_levels, single_dims, device):
        code_dims = [single_dims]*code_levels.shape[1]
        max_vals = list(np.max(code_levels, axis=0))
        code_levels = torch.from_numpy(code_levels).to(device)

        self.hier_embed_layer = HeirarchEmbedding(code_levels=code_levels, max_vals=max_vals, code_dims=code_dims)
        self.encoder = hslencoder()
        self.decoder = hsldecoder()



    def forward(self, x):
        return

    # def backward():
    #     return