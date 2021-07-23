# source: https://github.com/dido1998/Recurrent-Independent-Mechanisms

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np
import torch.multiprocessing as mp
import random

class blocked_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0


class AIM(object):
    def __init__(self, input_size, hidden_size, num_units, topk, 
                 input_key_size=64, input_value_size=400, input_query_size=64, num_input_heads=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_units = num_units
        self.key_size = input_key_size
        self.topk = topk
        self.num_input_heads = num_input_heads
        self.input_key_size = input_key_size
        self.input_query_size = input_query_size
        self.input_value_size = input_value_size

    def create_parameters(self):
        hs = nn.Parameter(torch.randn(self.num_units, self.hidden_size).cuda(), requires_grad=True)
        hs_weight = nn.Parameter(math.sqrt(2. / self.input_size) * torch.randn(self.num_units, self.input_size, self.input_value_size))
        query_weight = nn.Parameter(math.sqrt(2. / self.hidden_size) * torch.randn(self.num_units, self.hidden_size, self.input_query_size * self.num_input_heads))
        key_weight = nn.Parameter(torch.randn(self.num_input_heads * self.input_key_size, self.input_size))
        key_bias = nn.Parameter(torch.randn(self.num_input_heads * self.input_key_size))
        #value_weight = nn.Parameter(torch.randn(self.num_input_heads * self.input_value_size, self.input_size))
        #value_bias = nn.Parameter(torch.randn(self.num_input_heads * self.input_value_size))
        
        nn.init.normal_(key_weight, 0, math.sqrt(2. / self.input_size))
        nn.init.zeros_(key_bias)
        #nn.init.kaiming_normal_(value_weight)
        #nn.init.uniform_(value_bias, -1 / math.sqrt(self.input_size), 1 / math.sqrt(self.input_size))

        return [hs, hs_weight, query_weight, key_weight, key_bias]

    def grouplinearlayer(self, input, weight):
        input = input.permute(1, 0, 2)
        input = torch.bmm(input, weight)
        return input.permute(1, 0, 2)
                
    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def aim_forward_w_null(self, input, hs, hs_weight, query_weight, key_weight, key_bias, stochastic=None, mech_choice=None, threshold=True):
        input = input.unsqueeze(1)
        size = input.size()
        null_input = torch.zeros(size[0], 1, size[2]).float().cuda()
        input = torch.cat((input, null_input), dim = 1)
        key_layer = F.linear(input, key_weight, key_bias)
        query_layer = self.grouplinearlayer(hs.unsqueeze(0), query_weight)

        hs_value_layer = self.grouplinearlayer(input.unsqueeze(2).repeat(1, 1, self.num_units, 1).reshape(size[0] * 2, self.num_units, size[-1]), hs_weight)  # B*2 X num units X inval dim
        hs_value_layer = hs_value_layer.reshape(size[0], 2, self.num_units, self.input_value_size).permute(0, 2, 1, 3) # B X num units X 2 X inval dim

        key_layer = self.transpose_for_scores(key_layer, self.num_input_heads, self.input_key_size)
        query_layer = self.transpose_for_scores(query_layer, self.num_input_heads, self.input_query_size)
        #key_layer = F.normalize(key_layer, p=2, dim=-1, eps=1e-12)
        #query_layer = F.normalize(query_layer, p=2, dim=-1, eps=1e-12)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.input_key_size) 
        attention_scores = torch.mean(attention_scores, dim = 1)
        mask_ = torch.zeros(size[0], self.num_units).cuda()

        attention_scores = nn.Softmax(dim = -1)(attention_scores)
        #attention_scores = nn.Sigmoid()(attention_scores)
        not_null_scores = attention_scores[:, :, 0]

        # hard threshold
        if stochastic is None:
            topk1 = torch.topk(not_null_scores, self.topk, dim = 1)
        else:
            topk1 = torch.topk(not_null_scores, mech_choice, dim = 1).indices[:, stochastic]
        row_index = np.arange(size[0])
        row_index = np.repeat(row_index, self.topk)
        if stochastic:
            mask_[row_index, topk1.view(-1)] = 1
        else:
            mask_[row_index, topk1.indices.view(-1)] = 1
        if threshold:
            hs_value = torch.einsum('ijk,ijkl->ijl', 
                                    attention_scores[mask_.to(torch.bool)].reshape(size[0], -1, 2), 
                                    hs_value_layer[mask_.to(torch.bool)].reshape(size[0], -1, 2, hs_value_layer.size(3))) 
        else:
            hs_value = torch.einsum('ijk,ijkl->ijl', 
                                    attention_scores.reshape(size[0], -1, 2), 
                                    hs_value_layer.reshape(size[0], -1, 2, hs_value_layer.size(3))) 

        #hs_value = hs_value.sum(dim=1)
        #hs_value = F.normalize(hs_value, p=2, dim=hs_value.dim() - 1, eps=1e-12)
        #return hs_value, mask_, not_null_scores, attention_scores
        return hs_value.sum(dim=1), attention_scores, mask_

    def aim_forward(self, input, hs, hs_weight, query_weight, key_weight, key_bias, stochastic=None, mech_choice=None, threshold=True):
        input = input.unsqueeze(1)
        size = input.size()
        key_layer = F.linear(input, key_weight, key_bias)
        query_layer = self.grouplinearlayer(hs.unsqueeze(0), query_weight)

        hs_value_layer = self.grouplinearlayer(input.unsqueeze(2).repeat(1, 1, self.num_units, 1).reshape(size[0], self.num_units, size[-1]), hs_weight)  # B*2 X num units X inval dim
        hs_value_layer = hs_value_layer.reshape(size[0], 1, self.num_units, self.input_value_size).permute(0, 2, 1, 3) # B X num units X 2 X inval dim

        key_layer = self.transpose_for_scores(key_layer, self.num_input_heads, self.input_key_size)
        query_layer = self.transpose_for_scores(query_layer, self.num_input_heads, self.input_query_size)
        #key_layer = F.normalize(key_layer, p=2, dim=-1, eps=1e-12)
        #query_layer = F.normalize(query_layer, p=2, dim=-1, eps=1e-12)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.input_key_size) 
        attention_scores = torch.mean(attention_scores, dim = 1)
        mask_ = torch.zeros(size[0], self.num_units).cuda()

        #attention_scores = nn.Softmax(dim = -1)(attention_scores)
        #attention_scores = nn.Sigmoid()(attention_scores)
        not_null_scores = attention_scores[:, :, 0]

        # hard threshold
        if stochastic is None:
            topk1 = torch.topk(not_null_scores, self.topk, dim = 1)
        else:
            topk1 = torch.topk(not_null_scores, mech_choice, dim = 1).indices[:, stochastic]
        row_index = np.arange(size[0])
        row_index = np.repeat(row_index, self.topk)
        if stochastic:
            mask_[row_index, topk1.view(-1)] = 1
        else:
            mask_[row_index, topk1.indices.view(-1)] = 1
        if threshold:
            hs_value = torch.einsum('ijk,ijkl->ijl', 
                                    attention_scores[mask_.to(torch.bool)].reshape(size[0], -1, 1), 
                                    hs_value_layer[mask_.to(torch.bool)].reshape(size[0], -1, 1, hs_value_layer.size(3)))
        else:
            hs_value = torch.einsum('ijk,ijkl->ijl', 
                                    attention_scores.reshape(size[0], -1, 1), 
                                    hs_value_layer.reshape(size[0], -1, 1, hs_value_layer.size(3))) 

        #hs_value = hs_value.sum(dim=1)
        #hs_value = F.normalize(hs_value, p=2, dim=hs_value.dim() - 1, eps=1e-12)
        #return hs_value, mask_, not_null_scores, attention_scores

        #grad_mask = torch.ones(size=(self.topk,)).cuda()
        #grad_mask[:3] = 0
        #grad_mask.unsqueeze(0).repeat(size[0], 1)
        #hs_value = blocked_grad.apply(hs_value, grad_mask.unsqueeze(-1))
        return hs_value.sum(dim=1), attention_scores, topk1
