import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self, U, W, b, alpha):
        self.U = nn.Linear()
        self.W = nn.Linear()
        self.b = nn.Linear()
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        uw = torch.dot(self.U, self.W.t)
        ypred = self.alpha*self.softmax(uw + self.b)
        return ypred

    def backward():
        return