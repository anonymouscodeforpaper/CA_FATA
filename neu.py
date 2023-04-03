#!/usr/bin/env python
# coding: utf-8

# In[9]:


from torch import nn
from torch.nn import LeakyReLU
import torch
drop = torch.nn.Dropout(p=0.0)

# In[10]:


class NeuMF_contexts(nn.Module):
    def __init__(self, n_users, n_items, n_contexts,mlp_factors,gmf_factors, layers):
        super(NeuMF_contexts, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.mlp_factors = mlp_factors
        self.gmf_factors = gmf_factors
        self.layers = layers
        
        self.mlp_user_factors = torch.nn.Embedding(n_users, mlp_factors)
        self.mlp_item_factors = torch.nn.Embedding(n_items, mlp_factors)
        self.mlp_context_factors = torch.nn.Embedding(n_contexts, mlp_factors)
        
        self.gmf_user_factors = torch.nn.Embedding(n_users, gmf_factors)
        self.gmf_item_factors = torch.nn.Embedding(n_items, gmf_factors)
        
        self.MLP1 = nn.Sequential(nn.Linear(9 * mlp_factors, self.layers[0]))
        self.MLP2 = nn.Sequential(nn.Linear(self.layers[0],self.layers[1]))
        self.MLP3 = nn.Sequential(nn.Linear(self.layers[1],self.layers[2]))
        #self.MLP4 = nn.Sequential(nn.Linear(self.layers[2],self.layers[3]))
        
        self.outlayer = nn.Sequential(nn.Linear(self.layers[2] + gmf_factors,1))
        
        
    def forward(self, user_id, item_id,context_id):
        GMF_user_embeddings = self.gmf_user_factors(user_id) ## 128 * 64
        GMF_item_embeddings = self.gmf_item_factors(item_id) ## 128 * 64
        GMF_merge_embedding = GMF_user_embeddings * GMF_item_embeddings ## 128 * 64
        
        
        MLP_user_embeddings = self.mlp_user_factors(user_id) ## 128 * 64
        MLP_item_embeddings = self.mlp_item_factors(item_id) ## 128 * 64
        MLP_context_embeddings = self.mlp_user_factors(context_id)
        MLP_merge_embedding = torch.cat((MLP_user_embeddings,MLP_item_embeddings),1) ## 128 * 128
        MLP_context_embedding = torch.reshape(MLP_context_embeddings,(user_id.shape[0],-1))
        MLP_merge_embedding = torch.concat((MLP_merge_embedding,MLP_context_embedding),1)
        
        
        MLP_out1 = self.MLP1(MLP_merge_embedding) ## 128 * 64
        MLP_out1 = torch.nn.ReLU()(MLP_out1)
        
        MLP_out2 = self.MLP2(MLP_out1) ## 128 * 32
        MLP_out2 = torch.nn.ReLU()(MLP_out2)
        
        MLP_out3 = self.MLP3(MLP_out2) ## 128 * 16
        MLP_out3 = torch.nn.ReLU()(MLP_out3)
        
        #MLP_out4 = self.MLP4(MLP_out3) ## 128 * 8
        #MLP_out4 = torch.nn.ReLU()(MLP_out4)
        
        out = torch.cat((GMF_merge_embedding,MLP_out3),1)
        
        
        out = self.outlayer(out)  ## 128 * 1
        return out.squeeze(1)   


# In[11]:


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, mlp_factors,gmf_factors, layers):
        super(NeuMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.mlp_factors = mlp_factors
        self.gmf_factors = gmf_factors
        self.layers = layers
        
        self.mlp_user_factors = torch.nn.Embedding(n_users, mlp_factors)
        self.mlp_item_factors = torch.nn.Embedding(n_items, mlp_factors)
        
        self.gmf_user_factors = torch.nn.Embedding(n_users, gmf_factors)
        self.gmf_item_factors = torch.nn.Embedding(n_items, gmf_factors)
        
        self.MLP1 = nn.Sequential(nn.Linear(2 * mlp_factors, self.layers[0]))
        self.MLP2 = nn.Sequential(nn.Linear(self.layers[0],self.layers[1]))
        self.MLP3 = nn.Sequential(nn.Linear(self.layers[1],self.layers[2]))
        #self.MLP4 = nn.Sequential(nn.Linear(self.layers[2],self.layers[3]))
        
        self.outlayer = nn.Sequential(nn.Linear(self.layers[2] + gmf_factors,1))
        
        
    def forward(self, user_id, item_id):
        GMF_user_embeddings = self.gmf_user_factors(user_id) ## 128 * 64
        GMF_item_embeddings = self.gmf_item_factors(item_id) ## 128 * 64
        GMF_merge_embedding = GMF_user_embeddings * GMF_item_embeddings ## 128 * 64
        
        
        MLP_user_embeddings = self.mlp_user_factors(user_id) ## 128 * 64
        MLP_item_embeddings = self.mlp_item_factors(item_id) ## 128 * 64
        MLP_merge_embedding = torch.cat((MLP_user_embeddings,MLP_item_embeddings),1) ## 128 * 128
        
        
        MLP_out1 = self.MLP1(MLP_merge_embedding) ## 128 * 64
        MLP_out1 = torch.nn.ReLU()(MLP_out1)
        
        MLP_out2 = self.MLP2(MLP_out1) ## 128 * 32
        MLP_out2 = torch.nn.ReLU()(MLP_out2)
        
        MLP_out3 = self.MLP3(MLP_out2) ## 128 * 16
        MLP_out3 = torch.nn.ReLU()(MLP_out3)
        
        #MLP_out4 = self.MLP4(MLP_out3) ## 128 * 8
        #MLP_out4 = torch.nn.ReLU()(MLP_out4)
        
        out = torch.cat((GMF_merge_embedding,MLP_out3),1)
        
        
        out = self.outlayer(out)  ## 128 * 1
        return out.squeeze(1)   


# In[12]:


class MF(nn.Module):
    def __init__(self, n_users, n_items,n_factors):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
    def forward(self, user, item):
        u_final = self.user_factors(user).unsqueeze(1)
        i_final = self.item_factors(item).unsqueeze(1)
        u_final = drop(u_final)
        i_final = drop(i_final)
        return (u_final * i_final).sum(2).squeeze(1)


# In[ ]:




