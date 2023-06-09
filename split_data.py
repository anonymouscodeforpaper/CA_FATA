# -*- coding: utf-8 -*-
"""data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pnZaKJbOqRwvijELe8KbUxG9y1B0z3iN
"""



import pandas as pd
import numpy as np
import sklearn


def read_knowledge(args): ### Read attributes of items
    if args.name == 'Frappe':
        meta_app_knowledge = pd.read_csv('data/frappe/meta_app.csv')
        n_entities = max(meta_app_knowledge['rating']) + 1
        n_rel = meta_app_knowledge.shape[1] - 1
    if args.name == 'Yelp':
        meta_app_knowledge = pd.read_csv('data/yelp/df_business.csv')
        n_entities = max(meta_app_knowledge['att1']) + 1
        n_rel = meta_app_knowledge.shape[1] - 1
        
    return meta_app_knowledge, n_entities, n_rel

def read_context(args):  ### Read the interactions and split them into training, test and validation set
    if args.name == 'Frappe':
        df = pd.read_csv('data/frappe/final_app.csv')
        n_cf = df.shape[1] - 3 ## Here, we compute the number of contextual factors
        n_users = len(df['user'].value_counts()) ## Here, we compute the number of users
        n_items = len(df['item'].value_counts()) ## Here, we compute the number of items
        n_contexts = max(df['city']) + 1 ## Here, we compute the number of contextual conditions
        
        
        df_train = pd.read_csv('data/frappe/train.csv')

        
        df_test = pd.read_csv('data/frappe/test.csv')

        
        
        
        
    if args.name == 'Yelp':
        df = pd.read_csv('data/yelp/df_usable.csv')
        n_cf = df.shape[1] - 3 ## Here, we compute the number of contextual factors
        n_users = len(df['user'].value_counts()) ## Here, we compute the number of users
        n_items = len(df['item'].value_counts()) ## Here, we compute the number of items
        n_contexts = max(df['Alone_Companion']) + 1 ## Here, we compute the number of contextual conditions
        
        
        df_train = pd.read_csv('data/yelp/train.csv')
        df_train['cnt'] = (df_train['cnt'] - 1) / 2 - 1
        df_test = pd.read_csv('data/yelp/test.csv')
        df_test['cnt'] = (df_test['cnt'] - 1) / 2 - 1
    return df_train, df_test, n_users, n_items, n_contexts, n_cf
