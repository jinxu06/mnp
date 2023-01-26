import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time 
import typing
from tsalib import dim_vars as dvs, get_dim_vars, update_dim_vars_len
from tsalib import warp

from .mlp import MLP

_B, _S, _C, _T = dvs('Batch(b):10 SetSize(s):1 Context(c):1 Target(t):1', exists_ok=True)
_X, _Y, _Z, _R = dvs('DimX(x):1 DimY(y):1 DimZ(z):1 DimR(r):1', exists_ok=True)

# class BiDimensionalAttentionBlock(nn.Module):
    
#   def __init__(self,
#                dim_features: int = 128,
#                n_sabs: int = 3,
#                n_heads: int = 4):
#     super().__init__()
#     self.n_sabs = n_sabs 
#     self.row_sabs = [SAB(dim_features, dim_features, n_heads, ln=False) for _ in range(n_sabs)]
#     self.col_sabs = [SAB(dim_features, dim_features, n_heads, ln=False) for _ in range(n_sabs)]
#     self.row_sabs = nn.Sequential(*self.row_sabs)
#     self.col_sabs = nn.Sequential(*self.col_sabs)
    
#   def forward(self, 
#               X: torch.Tensor):
#     """
#     Args:
#         X: [batch_size, set_size, dim_x, dim_features]
#     """
#     update_dim_vars_len({'b': X.shape[0],
#                          's': X.shape[1],
#                          'x': X.shape[2],
#                          'r': X.shape[3]})
#     _B, _S, _C, _T, _X, _Y, _Z, _R = get_dim_vars('b s c t x y z r')
#     X_ = warp(X, 'b,s,x,r -> b*s,x,r', 'v')
#     X_col: 'b*s,x,r' = self.row_sabs(X_)
#     X_col = warp(X_col, 'b*s,x,r -> b,s,x,r', 'v')
#     X_ = warp(X, 'b,s,x,r -> b,x,s,r', 'p').clone()
#     X_ = X_.reshape(_B*_X,_S,_R)
#     X_row = self.row_sabs(X_)
#     X_row = warp(X_row, 'b*x,s,r -> b,x,s,r -> b,s,x,r', 'vp')
#     return X + torch.nn.SiLU()(X_row + X_col)


# class ResidualBiDimensionalSetEncoder(nn.Module):
    
#   def __init__(self,
#                dim_input_y: int,
#                dim_output: int,
#                dim_hidden: int = 128,
#                n_blocks: int = 2,
#                n_sabs: int = 2,
#                n_heads: int = 4,
#                aggregation: bool = True):
#     super().__init__()
#     self.n_blocks = n_blocks
#     self.aggregation = aggregation
#     self.dim_hidden = dim_hidden 
#     self.preprocessing = nn.Linear(dim_input_y+1, dim_hidden)
#     self.battns = [BiDimensionalAttentionBlock(dim_features=dim_hidden,
#                                         n_sabs=n_sabs,
#                                         n_heads=n_heads) for _ in range(n_blocks)]
#     self.battns = nn.Sequential(*self.battns)
#     if aggregation: 
#         self.pma = PMA(dim_hidden, n_heads, 1, ln=False)
#     self.linear = nn.Linear(dim_hidden, dim_output)
    
#   def forward(self,
#               X: torch.Tensor, 
#               Y: torch.Tensor):
#     """
#     Args:
#         X: [batch_size, set_size, dim_X]
#         Y: [batch_size, set_size, dim_Y]
#     """
#     update_dim_vars_len({'b': X.shape[0],
#                          's': X.shape[1],
#                          'x': X.shape[2],
#                          'y': Y.shape[2],
#                          'r': self.dim_hidden})
#     _B, _S, _C, _T, _X, _Y, _Z, _R = get_dim_vars('b s c t x y z r')
    
#     X = warp(X, 'b,s,x -> b,s,x,1', 'a')
#     Y = warp(Y, 'b,s,y -> b,s,1,y -> b,s,x,y', 'ar')
#     XY: 'b,s,x,y+1' = torch.cat([X, Y], dim=-1)
#     XY: 'b,s,x,r' = self.preprocessing(XY)
#     XY = self.battns(XY)
#     if self.aggregation:
#         XY = warp(XY, 'b,s,x,r -> b,x,s,r', 'p')
#         XY = XY.reshape(_B*_X,_S,_R)
#         XY = self.pma(XY)[:, 0]
#         XY = warp(XY, 'b*x,r -> b,x,r', 'v')
#     XY = self.linear(XY)
#     return XY

class DeepSetsEncoder(nn.Module):
    
  def __init__(self,
               dim_input: int,
               dim_output: int,
               dim_hidden: int = 128,
               n_layers: int = 3,
               aggregation: bool = True):
    super().__init__()
    self.dim_input = dim_input
    self.dim_output = dim_output 
    self.dim_hidden = dim_hidden
    self.n_layers = n_layers 
    self.aggregation = aggregation 
    
    self.shared_mlp = MLP(in_size=dim_input,
                          out_sizes=[dim_hidden] * n_layers)
    self.intermediate_layer = nn.Sequential(nn.Linear(dim_hidden, dim_output), nn.ReLU())
    self.iterative_mean = None 
    self.iterative_count = 0 
    
  def forward(self, 
              X: torch.Tensor,
              update=False):
    """
    Forwarding Set Encoder 
    Args:
        X: torch.Tensor, [batch_size, set_size, dim_input]
    Returns:
        torch.Tensor, [batch_size, set_size, dim_output] or [batch_size, dim_output]
    """
    if update and self.iterative_mean is not None:
      self.update_iterative_mean(X)
    else:
      B, S, D = X.shape 
      R = self.shared_mlp(X.view(B * S, D))
      if self.aggregation:  
        R = R.view(B, S, -1)
        self.iterative_mean = R.mean(dim=1)
        self.iterative_count = S
    return self.intermediate_layer(self.iterative_mean) # nn.ReLU()(self.linear(self.iterative_mean))
  
  def update_iterative_mean(self,
                            X_update: torch.Tensor):
    B, S, D = X_update.shape
    R_update = self.shared_mlp(X_update.view(B * S, D))
    R_update = R_update.view(B, S, -1)
    self.iterative_mean = self.iterative_mean * self.iterative_count + R_update.sum(dim=1)
    self.iterative_count += S
    self.iterative_mean = self.iterative_mean / self.iterative_count
    self.iterative_mean = self.iterative_mean.detach()
    

class SetTransformerEncoder(nn.Module):
    
  def __init__(self,
               dim_input: int,
               dim_output: int,
               dim_hidden: int = 128,
               n_sabs: int = 3,
               n_heads: int = 4,
               n_inds: int = 0,
               aggregation: bool = True):
    super().__init__()
    self.dim_input = dim_input
    self.dim_output = dim_output 
    self.dim_hidden = dim_hidden
    self.n_sabs = n_sabs
    self.n_heads = n_heads
    self.n_inds = n_inds 
    self.aggregation = aggregation 
    if n_inds <= 0:
        modules = [SAB(dim_input, dim_hidden, n_heads, ln=False)] \
                + [SAB(dim_hidden, dim_hidden, n_heads, ln=False) for _ in range(self.n_sabs)]
    else:
        modules = [ISAB(dim_input, dim_hidden, n_heads, n_inds, ln=False)] \
                + [ISAB(dim_hidden, dim_hidden, n_heads, n_inds, ln=False) for _ in range(self.n_sabs)]
    if aggregation:
      modules += [PMA(dim_hidden, n_heads, 1, ln=False)]
    modules += [nn.Linear(dim_hidden, dim_output), nn.ReLU()]
    self.enc = nn.Sequential(*modules)
    
  def forward(self, 
              X: torch.Tensor,
              update: bool = False):
    """
    Forwarding Set Encoder 
    Args:
        X: torch.Tensor, [batch_size, set_size, dim_input]
    Returns:
        torch.Tensor, [batch_size, set_size, dim_output] or [batch_size, dim_output]
    """
    if self.aggregation:
      return self.enc(X)[:, 0]
    else:
      return self.enc(X)
      
      
#######################################################################

class DimensionInvariantFeatureEncoder(nn.Module):
        
    def __init__(self, dim_input, num_outputs, dim_output,
            dim_hidden=128, num_heads=4, ln=False):
        super(DimensionInvariantFeatureEncoder, self).__init__()
        self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        s = list(X.shape)
        X = torch.flatten(X, start_dim=0, end_dim=-2).unsqueeze(-1)
        return self.enc(X).reshape(s[:-1] + [-1])
        

class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)

class CrossAttentionModule(nn.Module):
    
    def __init__(self, 
                 x_dim: int,
                 y_dim: int,
                 r_dim: int,
                 dim_hidden: int = 128,
                 num_heads: int = 1,
                 layer_norm: bool = False):
        super().__init__()
        self.sabs = nn.Sequential(
                SAB(x_dim+y_dim, dim_hidden, num_heads, ln=layer_norm),
                SAB(dim_hidden, dim_hidden, num_heads, ln=layer_norm),
                SAB(dim_hidden, r_dim, num_heads, ln=layer_norm))
        self.attn = ScaledDotProductAttention()

    def forward(self, 
                X_q: torch.Tensor, 
                X_c: torch.Tensor,
                Y_c: torch.Tensor):
        R_c = self.sabs(torch.cat([X_c, Y_c], dim=-1))
        R_q = self.attn(X_q, X_c, R_c)
        return R_q


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
