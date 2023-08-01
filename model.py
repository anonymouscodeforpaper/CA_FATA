# -*- coding: utf-8 -*-


import torch
from torch import nn
import torch
from torch import nn
from torch.nn import LeakyReLU
leaky = LeakyReLU(0.1)
drop = torch.nn.Dropout(p=0.4)
class aggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        #self.wei = torch.nn.Embedding(self.input_dim,self.output_dim)
        #self.bias = torch.nn.Embedding(1,self.output_dim)

    def forward(self, user, to_agg, rel):
        '''
        user: 128 * 64
        rel: 64 * 5
        to_agg: 128 * 5 * 64
        '''
        scores = torch.matmul(user, rel) # 128 * 5
        scores = leaky(scores)
        m = torch.nn.Softmax(dim=1) # 128 * 5
        scores = m(scores) # 128 * 5
        scores1 = scores.unsqueeze(1) #128 * 1 * 5
        context_agg = torch.bmm(scores1, to_agg) # 128 * 1 * 64
        return context_agg



class CA_FATA(nn.Module):
    def __init__(self, n_users, n_contexts, n_rc, n_entity, n_kc, n_factors, context_or, average_or):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.context_factors = torch.nn.Embedding(n_contexts, n_factors)
        self.relation_c = torch.nn.Embedding(n_factors, n_rc)
        self.entity_factors = torch.nn.Embedding(n_entity, n_factors)
        self.relation_k = torch.nn.Embedding(n_factors, n_kc) # 64 * 5
        self.agg = aggregator(n_factors, n_factors)
        self.context_or = context_or
        self.average_or = average_or

    def forward(self, user, contexts_index, entities_index):
        entities = self.entity_factors(entities_index)  # 128 * 5 * 64
        contexts = self.context_factors(contexts_index)  # 128 * 5 * 64

        if self.context_or:
            u_nei = self.agg(self.user_factors(user), contexts, self.relation_c.weight)  # 128 * 1 * 64
            u_final = u_nei + self.user_factors(user)[:, 0].unsqueeze(1).unsqueeze(1) # 128 * 1 * 64
            u_final = leaky(u_final)
        else:
            u_final = self.user_factors(user)

        if self.average_or:
            importances = torch.matmul(u_final.squeeze(1), self.relation_k.weight) # 128 * 5
            importances = leaky(importances)
            m = torch.nn.Softmax(dim=1) # 128 * 5
            importances = m(importances) # 128 * 5
            scores = torch.bmm(entities, u_final.squeeze(1).unsqueeze(2)) # 128 * 5 * 1
            scores_final = torch.bmm(importances.unsqueeze(1), scores) # 128 * 1 * 1
            scores_final.squeeze_(2) # 128 * 1
            scores_final.squeeze_(1) # 128
        else:
            scores = torch.bmm(entities, u_final.squeeze(1).unsqueeze(2))
            scores_final = scores.sum(1) / (entities_index.shape[1])
            scores_final.squeeze_(1)

        return scores_final

          
    

        
      

    
    
    

    
    
    
    
