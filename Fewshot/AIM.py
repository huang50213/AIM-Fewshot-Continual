# source: https://github.com/dido1998/Recurrent-Independent-Mechanisms

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np
import torch.multiprocessing as mp
import random

from numpy.random import default_rng

rng = default_rng()

class blocked_grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0


class GroupLinearLayer(nn.Module):

    def __init__(self, din, dout, num_blocks, topk=None):
        super(GroupLinearLayer, self).__init__()

        if topk is None:
            self.w	 = nn.Parameter(math.sqrt(2./din) * torch.randn(num_blocks,din,dout))
        else:
            self.w	 = nn.Parameter(math.sqrt(2./(din*topk)) * torch.randn(num_blocks,din,dout))

    def forward(self,x):
        x = x.permute(1,0,2)
        x = torch.bmm(x,self.w)
        return x.permute(1,0,2)


class AIM(nn.Module):
    def __init__(self, out_class, input_size, hidden_size, num_units, topk, input_key_size = 64, input_value_size = 400, input_query_size = 64,
        num_input_heads = 1, input_dropout = 0.1
    ):
        super().__init__()
        self.out_class = out_class
        self.hidden_size = hidden_size
        self.num_units = num_units
        self.key_size = input_key_size
        self.topk = topk
        self.num_input_heads = num_input_heads
        self.input_key_size = input_key_size
        self.input_query_size = input_query_size
        self.input_value_size = input_value_size
       
        self.hs = torch.nn.Parameter(torch.randn(self.num_units, self.hidden_size).cuda(), requires_grad=True)

        self.key = nn.Linear(input_size, num_input_heads * input_key_size)
        self.value = nn.Linear(input_size, num_input_heads * input_value_size)
        self.hs_value = GroupLinearLayer(input_size, input_value_size, self.num_units)
        self.query = GroupLinearLayer(hidden_size,  input_query_size * num_input_heads, self.num_units)
        self.input_dropout = nn.Dropout(p=input_dropout)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / (m.in_features)))
                
    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def input_attention_mask(self, x, h, eval=False, stochastic=None, mech_choice=None):
        """
        Input : x (batch_size, 2, input_size) [The null input is appended along the first dimension]
                h (batch_size, num_units, hidden_size)
        Output: inputs (list of size num_units with each element of shape (batch_size, input_value_size))
                mask_ binary array of shape (batch_size, num_units) where 1 indicates active and 0 indicates inactive
        """
        
        size = x.size()
        null_input = torch.zeros(size[0], 1, size[2]).float().cuda()
        x = torch.cat((x, null_input), dim = 1)
        key_layer = self.key(x)
        query_layer = self.query(h.unsqueeze(0))

        hs_value_layer = self.hs_value(x.unsqueeze(2).repeat(1,1,self.num_units,1).reshape(x.size(0)*2,self.num_units,x.size(-1)))  # B*2 X num units X inval dim
        hs_value_layer = hs_value_layer.reshape(x.size(0),2,self.num_units,self.input_value_size).permute(0,2,1,3) # B X num units X 2 X inval dim

        key_layer = self.transpose_for_scores(key_layer,  self.num_input_heads, self.input_key_size)
        query_layer = self.transpose_for_scores(query_layer, self.num_input_heads, self.input_query_size)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.input_key_size) 
        attention_scores = torch.mean(attention_scores, dim = 1)
        mask_ = torch.zeros(x.size(0), self.num_units).cuda()

        attention_scores = nn.Softmax(dim = -1)(attention_scores)
        not_null_scores = attention_scores[:,:, 0]

        # hard threshold
        if stochastic is None:
            topk1 = torch.topk(not_null_scores, self.topk,  dim = 1)
        else:
            topk1 = torch.topk(not_null_scores, mech_choice,  dim = 1).indices[:,stochastic]
        row_index = np.arange(x.size(0))
        row_index = np.repeat(row_index, self.topk)
        if stochastic:
            mask_[row_index, topk1.view(-1)] = 1
        else:
            mask_[row_index, topk1.indices.view(-1)] = 1

        hs_value = torch.einsum('ijk,ijkl->ijl', attention_scores[mask_.to(torch.bool)].reshape(x.size(0),-1,2), hs_value_layer[mask_.to(torch.bool)].reshape(x.size(0),-1,2,hs_value_layer.size(3))) 


        # soft threshold
        # if stochastic is None:
        #     mask_[not_null_scores>0.5] = 1
        # else:
        #     mask_[not_null_scores>0.5 - random.uniform(0,0.3)] = 1
        # hs_value = torch.einsum('ijk,ijkl->ijl', attention_scores*mask_.to(torch.bool).unsqueeze(-1), hs_value_layer*mask_.to(torch.bool).unsqueeze(-1).unsqueeze(-1)) 

        return hs_value, mask_, not_null_scores, attention_scores

    
    def forward(self, x, return_score=False, stochastic=None, eval=False, mech_choice=None):
        """
        Input : x (batch_size, 1 , input_size)
                hs (batch_size, num_units, hidden_size)
                cs (batch_size, num_units, hidden_size)
        Output: new hs, cs for LSTM
                new hs for GRU
        """
        
        hs_value, mask, score, att_score = self.input_attention_mask(x, self.hs, eval, stochastic, mech_choice)
        out = hs_value.sum(dim=1) # B X in_dim

        if return_score:
            return out, score, mask, att_score
        else:
            return out

