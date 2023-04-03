#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
import torch
from torch import nn
import time
from model import CA_FATA
from utilities import RMSE
from split_data import read_knowledge,read_context





def train_pre(args, verbos=False):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    meta_app_knowledge, n_entities, n_rel = read_knowledge(args)
    df_train, df_test, n_users, n_items, n_contexts, n_cf = read_context(args) ### Here, we get the training, test, and the validation set 
    
    model_val = CA_FATA(n_users, n_contexts, n_cf, n_entities, n_rel,args.dim,args.context_or,args.average_or).to(DEVICE)  ## We initialize the model
    
    
    
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model_val.parameters(), lr=args.lr,weight_decay = args.l2_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, threshold=0.05, threshold_mode='abs')
    
    
    
    
    
    for epoch in range(args.n_epochs):
        t1 = time.time()
        num_example = len(df_train)
        indices = list(range(num_example))
        for i in range(0, num_example, args.batch_size):
            indexs = indices[i:min(i+args.batch_size, num_example)]
            contexts_index = df_train.iloc[:, 3:].loc[indexs].values
            contexts_index = torch.tensor(contexts_index).to(DEVICE)
            item = df_train['item'].loc[indexs].values
            entities = meta_app_knowledge.loc[item].iloc[:, 1:].values
            entities_index = torch.tensor(entities).to(DEVICE)
            rating = torch.FloatTensor(
                df_train['cnt'].loc[indexs].values).to(DEVICE)
            user = df_train['user'].loc[indexs].values
            user = torch.LongTensor(user).to(DEVICE)
            prediction = model_val(user, contexts_index, entities_index)
            err = loss_func(prediction, rating)

                
            optimizer.zero_grad()

            err.backward()
            optimizer.step()
        t2 = time.time()
        rmse,mae, p,r,f,accuracy= RMSE(df_train,model_val,meta_app_knowledge,args)
        rmse1,mae1, p1,r1,f1,accuracy1= RMSE(df_test,model_val,meta_app_knowledge,args)
        scheduler.step(rmse1)
        
        if verbos == True:
            print("Epoch: ", epoch)
            print("Loss:",err)
            print(" RMSE in train set: ", rmse, " MAE in train set:", mae,"Precision in train set: ",p,"Recall in train set: ", r, "F1 score in train set: ", f)
            print(" RMSE in test set: ", rmse1, " MAE in test set:", mae1,"Precision in test set: ",p1,"Recall in test set: ", r1, "F1 score in test set: ", f1)
            print("Time consumed: ", t2 - t1)





