import torch 
import torch.nn
import torch.nn.functional as F
from torch_scatter import scatter
import pyro
from loss import shy_loss
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
        self.fc1 = nn.Linear(embedding_size * 2, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

    def forward(self, X, V, E):
        eX = scatter(X[V], E, dim=0, reduce='mean')
        expanded_X = X.unsqueeze(1).expand(X.shape[0], eX.shape[0], X.shape[-1])
        expanded_eX = eX.unsqueeze(0).expand(X.shape[0], eX.shape[0], eX.shape[-1])
        concatenated_input = torch.cat((expanded_X, expanded_eX), dim=-1)
        mask_prob = self.relu(self.fc1(concatenated_input))
        mask_prob = self.fc2(mask_prob).squeeze(-1)
        mask_prob = F.sigmoid(mask_prob)
        return mask_prob

class HSL2(nn.Module):
    def __init__(self, ns, c, ratio, temp):
        super().__init__()
        self.phi = nn.Parameter(torch.randn(ns, c))
        self.ratio = ratio
        self.temp = temp

    def forward(self, X, V, E, H, mask_prob):
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

        delH = torch.zeros_like(H)
        delH[r, c] = 1.0
        newH = H + delH

        incident_mask = pyro.distributions.RelaxedBernoulliStraightThrough(
            temperature=self.temp, probs=mask_prob.clamp(0.1, 0.9)
        ).rsample()

        if incident_mask.shape != newH.shape:
            incident_mask = incident_mask.expand_as(newH)

        newH *=incident_mask
        return newH

class HyperG(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.gru = nn.GRU(in_channel, out_channel)
        self.fc1 = nn.Linear(out_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, H):
        V = torch.nonzero(H)[:, 0]
        E = torch.nonzero(H)[:, 1]
        if E.shape[0] == 0:
            # Return zero tensor with proper shape
            return torch.zeros(self.gru.hidden_size, device=H.device)        
        # Get hyperedge embeddings by averaging node features
        edge_embeddings = scatter(X[V], E, dim=0, reduce='mean')
        # vembeddings = torch.matmul(phi.T, X)
        # hstates, _ = self.gru(vembeddings)
        # attn = self.fc1(hstates)
        # alpha = self.sigmoid(attn)
        # U = torch.sum(torch.matmul(alpha, hstates))
        edge_embeddings = edge_embeddings.unsqueeze(1)  # Add sequence dimension
        hstates, _ = self.gru(edge_embeddings)
        
        # Attention mechanism
        attn = self.fc1(hstates)
        alpha = self.sigmoid(attn)
        
        # Weighted sum
        U = torch.sum(alpha * hstates, dim=0)
        return U.squeeze()

class HSLEncoder(nn.Module):
    def __init__(self, code_dims, HGNN_dim, after_HGNN_dim, HGNN_layer_num, nhead, K, temperature, add_ratio, n_c, hid_state_dim, dropout, HGNN_model, device):
        super().__init__()
        self.HGNN_layer_num = HGNN_layer_num
        if HGNN_layer_num >= 0:
            self.hgnn = HGNN(sum(code_dims), HGNN_dim, after_HGNN_dim, HGNN_layer_num, nhead, dropout, HGNN_model, device)
        else:
            self.nohgnn = nn.Linear(sum(code_dims), after_HGNN_dim)

        self.K = K
        self.hsl1 = nn.ModuleList(HSL1(after_HGNN_dim) for _ in range(self.K))
        self.hsl2 = nn.ModuleList(HSL2(n_c, after_HGNN_dim, addr, temp) for temp, addr in zip(temperature, add_ratio))
        self.hyperG = nn.ModuleList(HyperG(after_HGNN_dim, hid_state_dim) for _ in range(self.K))

    def forward(self, X, H):
        V = torch.nonzero(H)[:, 0]
        E = torch.nonzero(H)[:, 1]

        if self.HGNN_layer_num >= 0:
            X1 = self.hgnn(X, V, E, H)
        else:
            X1 = F.leaky_relu(self.nohgnn(X))

        O = torch.stack([self.hsl1[i](X1, V, E) for i in range(self.K)])
        tp = torch.stack([self.hsl2[i](X1, V, E, H, O[i]) for i in range(self.K)])
        latent_tp = torch.stack([self.hyperG[i](X1, tp[i]) for i in range(self.K)])
        return tp, latent_tp, O

class decoderRNN(nn.Module):
    def __init__(self, hidden, output):
        super().__init__()
        self.gru = nn.GRU(hidden, hidden)
        self.final = nn.Linear(hidden, output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden, X):
        # out = torch.matmul(input, X).view(-1, 1)
        out = torch.matmul(input, X).unsqueeze(0).unsqueeze(0)
        out = F.relu(out)
        out, hid = self.gru(out, hidden)
        # out = self.sigmoid(self.final(out[0]))
        out = self.sigmoid(self.final(out.squeeze(0)))

        return out, hid

class HSLDecoder(nn.Module):
    def __init__(self, ldim, K, pdim, code_num, device):
        super().__init__()
        self.context = nn.Linear(ldim*K, pdim)
        self.reconst = decoderRNN(pdim, code_num)
        self.code_num = code_num
        self.device = device

    def forward(self, latent_tp, visit_len, H, X):
        # ltp = torch.reshape(latent_tp, (-1,)).view(-1, 1)
        ltp = latent_tp.flatten().unsqueeze(0)
        decoder = self.context(ltp).unsqueeze(0)

        Hreconstructed = torch.zeros(visit_len, self.code_num, device=self.device)
        target_tensor = H.T
        decoder_input = torch.zeros(self.code_num, device=self.device)
        for di in range(visit_len):
            output, decoder = self.reconst(decoder_input, decoder, X)
            Hreconstructed[di] = output
            decoder_input = target_tensor[di]
        return Hreconstructed.T

class attention(nn.Module):
    def __init__(self, in_channel, code_num, kdim, heads, K):
        super().__init__()
        self.key = nn.Linear(in_channel, kdim)
        self.query = nn.Linear(in_channel, kdim)
        self.value = nn.Linear(in_channel, kdim)

        self.multi_attn = nn.MultiheadAttention(kdim, heads)
        self.output = nn.Linear(kdim, 1, bias=False)

        self.classifier = nn.Linear(in_channel, code_num)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, ltp):
        keys = self.key(ltp)
        querys = self.query(ltp)
        values = self.value(ltp)

        attn, _ = self.multi_attn(querys, keys, values, need_weights=False)
        attn_tp = self.output(attn).squeeze(-1)
        alpha = self.softmax(attn_tp)
        pred = self.softmax(self.classifier(ltp))

        # final_pred = torch.sum(pred * alpha.unsqueeze(alpha, -1).expand(-1, -1, pred.shape[-1]), -2)
        final_pred = torch.sum(pred * alpha.unsqueeze(-1), dim=-2)
        return final_pred, alpha
        

class SHy(nn.Module):
    def __init__(self, code_levels, single_dim, HGNN_dim, after_HGNN_dim, HGNN_layer_num, nhead, num_TP, temperature, add_ratio, n_c, hid_state_dim, dropout, key_dim, SA_head, HGNN_model, device):
        super().__init__()
        max_vals = list(np.max(code_levels, axis=0))
        code_levels = torch.from_numpy(code_levels).to(device)
        code_dims = [single_dim] * code_levels.shape[1]
        self.hier_embed = HeirarchialEmbedding(code_levels, max_vals, code_dims)
        
        self.encoder = HSLEncoder(code_dims, HGNN_dim, after_HGNN_dim, HGNN_layer_num, nhead, num_TP, temperature, add_ratio, n_c, hid_state_dim, dropout, HGNN_model, device)

        self.decoder = HSLDecoder(hid_state_dim, num_TP, sum(code_dims), code_levels.shape[0], device)

        self.fc = attention(hid_state_dim, code_levels.shape[0], key_dim, SA_head, num_TP)


    def forward(self, Hs, visit_lens):
        # Hierarchical embedding for medical codes.
        X = self.hier_embed()
        # Obtain multiple temporal phenotypes (and latent representations) via HSL & Decoder.
        tp_list = []; latent_tp_list = []; recon_H_list = []
        for i in range(len(Hs)):
            tp, latent_tp, _ = self.encoder(X, Hs[i][:, 0:int(visit_lens[i])])
            tp_list.append(tp)
            latent_tp_list.append(latent_tp)
            recon_H_list.append(self.decoder(latent_tp, int(visit_lens[i]), Hs[i], X))
        # Classify based on the temporal phenotype embeddings.
        pred, alphas = self.fc(torch.stack(latent_tp_list))
        return pred, tp_list, recon_H_list, alphas
