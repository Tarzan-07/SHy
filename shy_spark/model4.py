import torch 
import torch.nn
import torch.nn.functional as F
from torch_scatter import scatter
import pyro

import numpy as np 
from layers import *

class HeirarchialEmbedding(nn.Module):
    def __init__(self, code_levels, max_vals, code_dims):
        super().__init__()
        self.L = len(max_vals)
        self.code_levels = code_levels
        self.levelEmbeddings = nn.ModuleList(nn.Embedding(l, cd) for l, cd in zip(max_vals, code_dims))

    def forward(self):
        # embeddings = [self.levelEmbeddings[i](self.code_levels[:, i]-1) for i in range(self.L)]
        embeddings = []
        for i in range(self.L):
            embeddings.append(self.levelEmbeddings[i](self.code_levels[:, i] - 1))
        embeddings = torch.cat(embeddings, dim=1)
        return embeddings

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

class HSL1(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.fc1 = nn.Linear(embedding_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

    def forward(self, X, V, E):
        eX = scatter(X[V], E, dim=0, reduce='mean')
        expanded_X = torch.unsqueeze(1).expand(X.shape[0], eX.shape[0], X.shape[-1])
        expanded_eX = eX.repeat(X.shape[0], 1, 1)
        concatenated_input = torch.cat(expanded_X, expanded_eX)
        mask_prob = self.relu(self.fc1(concatenated_input))
        mask_prob = F.sigmoid(torch.squeeze(self.fc2(mask_prob)))
        return mask_prob

class HSL2(nn.Module):
    def __init__(self, ns, c, ratio, temp):
        super().__init__()
        self.phi = nn.Parameter(torch.randn(ns, c))
        self.ratio = ratio
        self.temp = temp

    def forward(self,  X, H, V, E, mask_prob):
        eX = scatter(X[V], E, dim=0, reduce='mean')
        a = X.unsqueeze(1) * self.phi
        b = eX.unsqueeze(1) * self.phi

        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)

        at = a.permute(1, 0, 2)
        bt = b.permute(1, 2, 0)

        sheads = torch.matmul(at, bt)
        S = sheads.mean(0)

        S[V, E] = -1e-30
        _, i = torch.topk(S.flatten(), int(self.ratio*E.shape[0]))
        r = torch.div(i, S.shape[1], rounding_mode='floor')
        c = i % S.shape[1]

        delH = torch.zeros(H)
        delH[r, c] = 1.0
        newH = H + delH

        incident_mask = pyro.distributions.RelaxedBernoulliStriaghtThrough(
            temperature=self.temp, probs=mask_prob
        ).rsample()
        newH *=incident_mask
        return newH

class HyperG(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.gru = nn.GRU(in_channel, out_channel)
        self.fc1 = nn.Linear(out_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, phi, X):
        vembeddings = torch.matmul(phi.T, X)
        hstates, _ = self.gru(vembeddings)
        attn = self.fc1(hstates)
        alpha = self.sigmoid(attn)
        U = torch.sum(torch.matmul(alpha, hstates))
        return U

class HSLEncoder(nn.Module):
    def __init__(self, code_dims, HGNN_dim, after_HGNN_dim, HGNN_layer_num, nhead, K, temperature, add_ratio, n_c, hid_state_dim, dropout, HGNN_model, device):
        super().__init__()
        self.HGNN_layer_num = HGNN_layer_num
        if HGNN_layer_num >= 0:
            self.firstHGNN = HGNN(sum(code_dims), HGNN_dim, after_HGNN_dim, HGNN_layer_num, nhead, dropout, HGNN_model, device)
        else:
            self.NoneHGNN = nn.Linear(sum(code_dims), after_HGNN_dim)

        self.K = K
        self.hsl1 = nn.ModuleList(HSL1(after_HGNN_dim) for _ in range(self.K))
        self.hsl2 = nn.ModuleList(HSL2(n_c, after_HGNN_dim, addr, temp) for temp, addr in zip(temperature, add_ratio))
        self.hyperG = HyperG(after_HGNN_dim, hid_state_dim)

    def forward(self):
        pass


class SHy(nn.Module):
    def __init__(self, code_levels, device, single_dim):
        super().__init__()
        max_vals = list(np.max(code_levels, axis=0))
        code_levels = torch.from_numpy(code_levels).to(device)
        code_dims = [single_dim] * code_levels.shape[1]
        self.hier_embed = HeirarchialEmbedding(code_levels, max_vals, code_dims)

    def forward(self):
        return