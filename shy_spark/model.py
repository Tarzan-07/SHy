import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


class shy(nn.Module):
    def __init__(self, code_levels, single_dims, device):
        code_dims = [single_dims]*code_levels.shape[1]
        max_vals = list(np.max(code_levels, axis=0))
        code_levels = torch.from_numpy(code_levels).to(device)

        self.hier_embed_layer = HeirarchEmbedding(code_levels=code_levels, max_vals=max_vals, code_dims=code_dims)



    def forward(self, x):
        uw = torch.dot(self.U, self.W.t)
        ypred = self.alpha*self.softmax(uw + self.b)
        return ypred

    def backward():
        return