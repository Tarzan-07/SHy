# shy_rewrite.py
import pyro
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from layers import *  # keep existing conv implementations available


# -------------------------
# Hierarchical code embedder
# -------------------------
class CodeHierarchyEmbed(nn.Module):
    """Create concatenated embeddings from multi-level code indices.

    Input:
      - levels: (num_codes, num_levels) torch tensor (1-indexed indices)
      - counts_per_level: list of ints = max index per level
      - dim_per_level: list of ints = embedding dim per level

    Output:
      - embeddings: (num_codes, sum(dim_per_level))
    """
    def __init__(self, levels: torch.Tensor, counts_per_level: list, dim_per_level: list):
        super().__init__()
        self.levels = levels
        self.num_levels = len(counts_per_level)

        # create a dedicated embedding table for each hierarchy level
        tables = []
        for max_id, d in zip(counts_per_level, dim_per_level):
            tables.append(nn.Embedding(num_embeddings=max_id, embedding_dim=d))
        self.tables = nn.ModuleList(tables)

    def forward(self):
        pieces = []
        # levels are 1-indexed in preprocessing; convert to 0-index before lookup
        for lvl in range(self.num_levels):
            idx = (self.levels[:, lvl] - 1).long()
            pieces.append(self.tables[lvl](idx))
        return torch.cat(pieces, dim=-1)


# -------------------------
# Hypergraph GNN wrapper
# -------------------------
class HypergraphBackbone(nn.Module):
    """Wrapper that instantiates a chosen hypergraph conv stack and runs it.

    Args mirror the original HGNN:
      - in_dim, hid_dim, out_dim, n_layers, n_heads, dropout, model_name, device
    """
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, n_heads, dropout, model_name, device):
        super().__init__()
        self.n_layers = n_layers
        self.model_name = model_name

        # create convolutional blocks per selected hypergraph conv type
        if model_name == "UniGINConv":
            first = UniGINConv(in_dim, hid_dim, heads=n_heads, dropout=0.)
            rest = [UniGINConv(hid_dim * n_heads, hid_dim, heads=n_heads, dropout=0.) for _ in range(max(0, n_layers-1))]
            self.layers = nn.ModuleList([first] + rest)
            if n_layers > 0:
                self.out_conv = UniGINConv(hid_dim * n_heads, out_dim, heads=1, dropout=0.)
            else:
                self.out_conv = UniGINConv(in_dim, out_dim, heads=1, dropout=0.)

        elif model_name == "UniSAGEConv":
            first = UniSAGEConv(in_dim, hid_dim, heads=n_heads, dropout=0.)
            rest = [UniSAGEConv(hid_dim * n_heads, hid_dim, heads=n_heads, dropout=0.) for _ in range(max(0, n_layers-1))]
            self.layers = nn.ModuleList([first] + rest)
            if n_layers > 0:
                self.out_conv = UniSAGEConv(hid_dim * n_heads, out_dim, heads=1, dropout=0.)
            else:
                self.out_conv = UniSAGEConv(in_dim, out_dim, heads=1, dropout=0.)

        elif model_name == "UniGATConv":
            first = UniGATConv(in_dim, hid_dim, heads=n_heads, dropout=0.)
            rest = [UniGATConv(hid_dim * n_heads, hid_dim, heads=n_heads, dropout=0.) for _ in range(max(0, n_layers-1))]
            self.layers = nn.ModuleList([first] + rest)
            if n_layers > 0:
                self.out_conv = UniGATConv(hid_dim * n_heads, out_dim, heads=1, dropout=0.)
            else:
                self.out_conv = UniGATConv(in_dim, out_dim, heads=1, dropout=0.)

        elif model_name == "UniGCNConv":
            first = UniGCNConv(in_dim, hid_dim, heads=n_heads, dropout=0.)
            rest = [UniGCNConv(hid_dim * n_heads, hid_dim, heads=n_heads, dropout=0.) for _ in range(max(0, n_layers-1))]
            self.layers = nn.ModuleList([first] + rest)
            if n_layers > 0:
                self.out_conv = UniGCNConv(hid_dim * n_heads, out_dim, heads=1, dropout=0.)
            else:
                self.out_conv = UniGCNConv(in_dim, out_dim, heads=1, dropout=0.)

        elif model_name == "UniGCNIIConv":
            # GCNII expects a prelude/postlude
            self.prelude = nn.Linear(in_dim, hid_dim)
            self.layers = nn.ModuleList([UniGCNIIConv(hid_dim, hid_dim, heads=n_heads, dropout=0.)] +
                                        [UniGCNIIConv(hid_dim, hid_dim, heads=n_heads, dropout=0.) for _ in range(max(0, n_layers-1))])
            if n_layers > 0:
                self.out_conv = UniGCNIIConv(hid_dim, hid_dim, heads=1, dropout=0.)
                self.postlude = nn.Linear(hid_dim, out_dim)
            else:
                self.out_conv = UniGCNIIConv(in_dim, in_dim, heads=1, dropout=0.)
                self.postlude = nn.Linear(in_dim, out_dim)

        elif model_name == "AllDeepSets":
            self.layers = nn.ModuleList([AllSet(in_dim, hid_dim, heads=n_heads, aggr='add', PMA=False, device=device, dropout=dropout)] +
                                        [AllSet(hid_dim, hid_dim, heads=n_heads, aggr='add', PMA=False, device=device, dropout=dropout) for _ in range(max(0, n_layers-1))])
            if n_layers > 0:
                self.out_conv = AllSet(hid_dim, out_dim, heads=n_heads, aggr='add', PMA=False, device=device, dropout=dropout)
            else:
                self.out_conv = AllSet(in_dim, out_dim, heads=n_heads, aggr='add', PMA=False, device=device, dropout=dropout)

        elif model_name == "AllSetTransformer":
            self.layers = nn.ModuleList([AllSet(in_dim, hid_dim, heads=n_heads, aggr='mean', PMA=True, device=device, dropout=dropout)] +
                                        [AllSet(hid_dim, hid_dim, heads=n_heads, aggr='mean', PMA=True, device=device, dropout=dropout) for _ in range(max(0, n_layers-1))])
            if n_layers > 0:
                self.out_conv = AllSet(hid_dim, out_dim, heads=n_heads, aggr='mean', PMA=True, device=device, dropout=dropout)
            else:
                self.out_conv = AllSet(in_dim, out_dim, heads=n_heads, aggr='mean', PMA=True, device=device, dropout=dropout)

        elif model_name == "HyperGCNConv":
            self.layers = nn.ModuleList([HyperGCNConv(in_dim, hid_dim, True, device, dropout)] +
                                        [HyperGCNConv(in_dim, hid_dim, True, device, dropout) for _ in range(max(0, n_layers-1))])
            if n_layers > 0:
                self.out_conv = HyperGCNConv(hid_dim, out_dim, True, device, dropout)
            else:
                self.out_conv = HyperGCNConv(in_dim, out_dim, True, device, dropout)

        else:
            raise ValueError(f"Unknown hypergraph model: {model_name}")

        self.activation = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, X, V, E, H):
        # Note: some convs expect (X, V, E, H) others (X, V, E) - preserve call signatures
        if self.model_name == "UniGCNConv":
            if self.n_layers > 0:
                for conv in self.layers:
                    X = conv(X, V, E, H)
                    X = self.activation(X)
                    X = self.drop(X)
            X = self.out_conv(X, V, E, H)

        elif self.model_name == "UniGCNIIConv":
            if self.n_layers > 0:
                X = F.relu(self.prelude(X))
                X0 = X
                for conv in self.layers:
                    X = conv(X, V, E, X0, H)
                    X = self.activation(X)
                    X = self.drop(X)
                X = self.out_conv(X, V, E, X0, H)
                X = self.postlude(X)
            else:
                X = self.out_conv(X, V, E, X, H)
                X = self.postlude(X)

        else:
            if self.n_layers > 0:
                for conv in self.layers:
                    X = conv(X, V, E)
                    X = self.activation(X)
                    X = self.drop(X)
            X = self.out_conv(X, V, E)

        return F.leaky_relu(X)


# -------------------------
# HSL Part 1: node-visit probability network
# -------------------------
class NodeVisitProb(nn.Module):
    """Compute soft probability that node i belongs to visit j.

    Produces a (num_nodes, num_visits) probability matrix.
    """
    def __init__(self, feat_dim, hidden=256):
        super().__init__()
        self.pair_mlp = nn.Linear(feat_dim * 2, hidden)
        self.nonlinear = nn.ReLU()
        self.out = nn.Linear(hidden, 1)

    def forward(self, node_feats, node_indices, visit_indices):
        # produce visit-level vectors by averaging node features within each visit
        visit_feats = scatter(node_feats[node_indices], visit_indices, dim=0, reduce='mean')

        # prepare a (num_nodes, num_visits, 2*feat_dim) tensor of all pairs (node, visit)
        n_nodes = node_feats.shape[0]
        n_visits = visit_feats.shape[0]

        nodes_expanded = node_feats.unsqueeze(1).expand(n_nodes, n_visits, node_feats.shape[-1])
        visits_tiled = visit_feats.repeat(n_nodes, 1, 1)

        pair_input = torch.cat([nodes_expanded, visits_tiled], dim=-1)
        h = self.nonlinear(self.pair_mlp(pair_input))
        probs = torch.sigmoid(torch.squeeze(self.out(h)))  # shape might be (N, T) or (N*T,) -> squeeze
        if probs.dim() == 1:
            probs = probs.unsqueeze(1)
        return probs


# -------------------------
# HSL Part 2: false-negative addition + sampling
# -------------------------
class FalseNegativeAndSampler(nn.Module):
    """Add likely missing node→visit links (false negatives) and sample nodes with RelaxedBernoulli.

    Arguments:
      - n_codes: number of disease codes (channels for cos_weight)
      - feat_dim: feature dimension
      - add_ratio: fraction of edges to add (top-k)
      - temp: temperature for RelaxedBernoulli
    """
    def __init__(self, n_codes, feat_dim, add_ratio, temp):
        super().__init__()
        # learnable per-code projection vectors (as in original implementation)
        self.proj = nn.Parameter(torch.randn(n_codes, feat_dim))
        self.add_ratio = add_ratio
        self.temp = temp

    def forward(self, node_feats, H, V, E, incident_prob):
        # compute visit means
        visit_feats = scatter(node_feats[V], E, dim=0, reduce='mean')

        # project nodes and visits and compute similarity S
        node_proj = node_feats.unsqueeze(1) * self.proj  # shape: (N, C, F) where C = n_codes
        node_norm = F.normalize(node_proj, p=2, dim=-1).permute(1, 0, 2)  # (C, N, F)

        visit_proj = visit_feats.unsqueeze(1) * self.proj
        visit_norm = F.normalize(visit_proj, p=2, dim=-1).permute(1, 2, 0)  # (C, F, T)

        S = torch.matmul(node_norm, visit_norm).mean(0)  # (N, T) similarity (averaged over channels)

        # mask out existing edges so we don't pick them as "new"
        S[V, E] = -1e30

        # pick top-k positions to add false negatives
        k = int(self.add_ratio * E.shape[0])
        if k > 0:
            vals, idxs = torch.topk(S.flatten(), k)
            rows = torch.div(idxs, S.shape[1], rounding_mode="floor")
            cols = idxs % S.shape[1]
            delta = torch.zeros_like(H)
            delta[rows, cols] = 1.0
        else:
            delta = torch.zeros_like(H)

        enriched = H + delta

        # sample node inclusion with RelaxedBernoulli (straight-through)
        sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temp, probs=incident_prob).rsample()
        enriched = enriched * sampled

        return enriched


# -------------------------
# Aggregator: temporal GRU + attention
# -------------------------
class TemporalPhenotypeAggregator(nn.Module):
    """Aggregate visits into a single phenotype vector via GRU + attention."""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers=1)
        self.att_ctx = nn.Linear(hidden_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, node_feats, H):
        # H: (num_nodes, num_visits) -> compute visit embeddings V_j = H^T @ X  (visit count, feat)
        visit_emb = torch.matmul(H.T.to(torch.float32), node_feats)  # (T, F)
        visit_emb = visit_emb.unsqueeze(1) if visit_emb.dim() == 1 else visit_emb  # ensure (T, F)

        # run GRU to get hidden sequence (T, hidden)
        # GRU expects (seq_len, batch, input_size), we want seq_len=T, batch=1
        seq = visit_emb  # shape (T, feat)
        seq_in = seq.unsqueeze(1)  # (T, 1, feat)
        hidden_states, _ = self.gru(seq_in)  # (T, 1, hidden)
        hidden_states = hidden_states.squeeze(1)  # (T, hidden)

        # temporal attention
        scores = torch.squeeze(self.att_ctx(hidden_states), dim=-1)  # (T,)
        alpha = self.softmax(scores)  # (T,)
        # weighted sum of hidden states
        weighted = torch.matmul(alpha.unsqueeze(0), hidden_states)  # (1, hidden)
        phenotype_vec = weighted.squeeze(0)  # (hidden,)
        return phenotype_vec


# -------------------------
# HSL Encoder (ties everything)
# -------------------------
class PhenotypeEncoder(nn.Module):
    """Encapsulates HGNN + HSL parts to output per-phenotype incidence matrices and latent vectors."""
    def __init__(self, code_dims, hgnn_in, hgnn_out, hgnn_layers, n_heads, num_TP, temps, add_ratios, code_count, phenotype_dim, dropout, hgnn_model, device):
        super().__init__()
        self.hgnn_layers = hgnn_layers
        if hgnn_layers >= 0:
            self.hgnn = HypergraphBackbone(sum(code_dims), hgnn_in, hgnn_out, hgnn_layers, n_heads, dropout, hgnn_model, device)
        else:
            self.identity_map = nn.Linear(sum(code_dims), hgnn_out)

        self.num_TP = num_TP
        self.part1_nets = nn.ModuleList([NodeVisitProb(hgnn_out) for _ in range(num_TP)])
        self.part2_nets = nn.ModuleList([FalseNegativeAndSampler(code_count, hgnn_out, add, temp) for temp, add in zip(temps, add_ratios)])
        self.aggregator = TemporalPhenotypeAggregator(hgnn_out, phenotype_dim)

    def forward(self, X, H):
        # Build sparse edge lists V, E from incidence matrix H
        V = torch.nonzero(H)[:, 0]
        E = torch.nonzero(H)[:, 1]

        # update node features via HGNN if present
        if self.hgnn_layers >= 0:
            X1 = self.hgnn(X, V, E, H)
        else:
            X1 = F.leaky_relu(self.identity_map(X))

        if self.num_TP > 1:
            # compute per-phenotype incident probabilities
            incident_probs = torch.stack([m(X1, V, E) for m in self.part1_nets])
            # apply false-negative addition + sampling --> per-phenotype incidence matrices
            TPs = torch.stack([self.part2_nets[k](X1, H, V, E, incident_probs[k]) for k in range(self.num_TP)])
            # aggregate each phenotype graph into a latent vector
            latent = torch.stack([self.aggregator(X1, TPs[k]) for k in range(self.num_TP)])
        else:
            incident_probs = self.part1_nets[0](X1, V, E)
            TPs = self.part2_nets[0](X1, H, V, E, incident_probs)
            latent = self.aggregator(X1, TPs)

        return TPs, latent, incident_probs


# -------------------------
# Decoder (reconstruct visits from latent phenotypes)
# -------------------------
class SmallGRUDecoder(nn.Module):
    def __init__(self, hidden, out_dim):
        super().__init__()
        self.gru = nn.GRU(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)
        self.act_sig = nn.Sigmoid()

    def forward(self, inp, hidden, code_embeds):
        # inp: (code_num,) vector used to form input, code_embeds used for mapping
        out = F.relu(torch.matmul(inp, code_embeds).view(1, -1))
        out, hidden = self.gru(out, hidden)
        out = self.act_sig(self.out(out[0]))
        return out, hidden


class PhenotypeDecoder(nn.Module):
    def __init__(self, latent_dim, num_TP, proj_dim, code_count, device):
        super().__init__()
        self.to_ctx = nn.Linear(latent_dim * num_TP, proj_dim)
        self.decoder_block = SmallGRUDecoder(proj_dim, code_count)
        self.device = device
        self.code_count = code_count

    def forward(self, latent_tp, visit_len, H, code_embeddings):
        # latent_tp: either (num_TP, hidden) or (hidden,) depending on num_TP
        ctx = self.to_ctx(torch.reshape(latent_tp, (-1,))).view(1, -1)  # initial hidden state
        rec_H = torch.zeros(visit_len, self.code_count, device=self.device)
        target = H.T  # shape (T, codes)
        decoder_in = torch.zeros(self.code_count, device=self.device)
        hidden = ctx
        for t in range(visit_len):
            out, hidden = self.decoder_block(decoder_in, hidden, code_embeddings)
            rec_H[t] = out[0]
            decoder_in = target[t]
        return rec_H.T


# -------------------------
# Final classifier (attention over phenotypes)
# -------------------------
class TPClassifier(nn.Module):
    def __init__(self, in_dim, code_count, key_dim, n_heads, num_TP):
        super().__init__()
        self.num_TP = num_TP
        if num_TP > 1:
            self.to_k = nn.Linear(in_dim, key_dim)
            self.to_q = nn.Linear(in_dim, key_dim)
            self.to_v = nn.Linear(in_dim, key_dim)
            self.mha = nn.MultiheadAttention(embed_dim=key_dim, num_heads=n_heads)
            self.tp_score = nn.Linear(key_dim, 1, bias=False)

        self.final_lin = nn.Linear(in_dim, code_count)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, latent_tp):
        # latent_tp: shape (num_TP, hidden) or (batch, num_TP, hidden) depending on caller
        if self.num_TP > 1:
            keys = self.to_k(latent_tp)
            queries = self.to_q(latent_tp)
            vals = self.to_v(latent_tp)
            sa_out, _ = self.mha(queries, keys, vals, need_weights=False)
            alpha = self.softmax(torch.squeeze(self.tp_score(sa_out), -1))
            per_tp_pred = self.softmax(self.final_lin(latent_tp))
            # combine per-TP predictions weighted by alpha
            weighted = torch.sum(per_tp_pred * torch.unsqueeze(alpha, -1).expand(-1, -1, per_tp_pred.shape[-1]), dim=-2)
            return weighted, alpha
        else:
            final_pred = self.softmax(self.final_lin(latent_tp))
            return final_pred, torch.rand(4)


# -------------------------
# Full SHy model assembly
# -------------------------
class SHyModel(nn.Module):
    def __init__(self, code_levels, base_dim, hgnn_dim, after_hgnn_dim, hgnn_layers, n_heads, num_TP,
                 temps, add_ratios, code_count_channel, phenotype_hidden, dropout, key_dim, sa_heads, hgnn_model, device):
        super().__init__()
        # code_levels: np array (num_codes, num_levels)
        counts_per_level = (np.max(code_levels, axis=0)).tolist()
        # convert to tensor for the embedder
        levels_tensor = torch.from_numpy(code_levels).to(device)
        level_dims = [base_dim] * levels_tensor.shape[1]

        self.code_embed = CodeHierarchyEmbed(levels_tensor, counts_per_level, level_dims)
        self.encoder = PhenotypeEncoder(level_dims, hgnn_dim, after_hgnn_dim, hgnn_layers, n_heads,
                                        num_TP, temps, add_ratios, code_count_channel, phenotype_hidden, dropout, hgnn_model, device)
        self.decoder = PhenotypeDecoder(phenotype_hidden, num_TP, sum(level_dims), levels_tensor.shape[0], device)
        self.classifier = TPClassifier(phenotype_hidden, levels_tensor.shape[0], key_dim, sa_heads, num_TP)

    def forward(self, Hs, visit_lens):
        code_X = self.code_embed()  # (num_codes, embed_dim)
        tp_collections = []
        latent_collection = []
        recon_collection = []
        for i in range(len(Hs)):
            real_H = Hs[i][:, 0:int(visit_lens[i])]
            tp, latent_tp, _ = self.encoder(code_X, real_H)
            tp_collections.append(tp)
            latent_collection.append(latent_tp)
            recon_collection.append(self.decoder(latent_tp, visit_lens[i], Hs[i], code_X))

        preds, alphas = self.classifier(torch.stack(latent_collection))
        return preds, tp_collections, recon_collection, alphas
