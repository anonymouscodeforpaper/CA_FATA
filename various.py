#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import f1_score, roc_auc_score,accuracy_score


# In[ ]:


def acc_auc(data,model,meta_app_knowledge):
    contexts_index = data.iloc[:, 3:].values
    contexts_index = torch.tensor(contexts_index).to(DEVICE)
    item = data['item'].values
    entities_index = meta_app_knowledge.loc[item].iloc[:, 1:].values
    entities_index = torch.tensor(entities_index).to(DEVICE)

    rating = torch.FloatTensor(data['cnt'].values).to(DEVICE)
    user = torch.LongTensor(data['user'].values).to(DEVICE)
    item = torch.LongTensor(data['item'].values).to(DEVICE)
    
    scores = model(user, item,contexts_index,entities_index)
    rating = rating.cpu().detach()
    scores = scores.cpu().detach()
    auc = roc_auc_score(rating, scores)
    scores[scores > 0.5] = 1
    scores[scores <= 0.5] = 0
    accuracy = accuracy_score(rating,scores)
    f1 = f1_score(y_true=rating, y_pred=scores)
    return accuracy, f1, auc


num_i = 101
def get_prediction_list_fata(model,i,user, target, contexts_index,entities_index,k):
    item_neg = user_item_neg[(i,user)]
    movie_rec = meta_app_knowledge[meta_app_knowledge['item'].isin(item_neg)]
    entities_o = movie_rec.iloc[:, 1:].values
    entities = np.insert(entities_o,1,values=entities_index,axis=0)
    
    items_o = movie_rec['item'].values
    items = np.insert(items_o,1,np.array(target))
    
    user = np.array([user] * num_i)
    user = torch.LongTensor(user).to(DEVICE)
    entities = torch.LongTensor(entities).to(DEVICE)
    contexts = np.tile(contexts_index,(num_i,1))
    contexts = torch.LongTensor(contexts).to(DEVICE)
    prediction,_,_ = model(user, contexts, entities)
    prediction = prediction.detach().numpy()
    prediction_dict = dict(zip(items, prediction))
    top_k = dic_order_value_and_get_key(prediction_dict,k)
    return top_k,prediction_dict

def hr_ndcg_fata(data,model):
    hr1 = 0.0
    ndcg1 = 0.0
    hr2 = 0.0
    ndcg2 = 0.0
    test_eva_dict = df_empty.to_dict('records')
    i = 0
    for row in test_eva_dict[:]:
        user = row['user']
        target = row['item']
        contexts_index = []
        contexts_index.append(row['daytime'])
        contexts_index.append(row['weekday'])
        contexts_index.append(row['isweekend'])
        contexts_index.append(row['homework'])
        contexts_index.append(row['weather'])
        contexts_index.append(row['country'])
        contexts_index.append(row['city'])
        contexts_index = np.array(contexts_index)
        entities_index = []
        
        entities_index.append(row['category'])
        entities_index.append(row['downloads'])
        entities_index.append(row['language'])
        entities_index.append(row['price'])
        entities_index.append(row['rating'])
        entities_index = np.array(entities_index)
        
        
        recommended_item_10,_ = get_prediction_list_fata(model,i,user,target,contexts_index,entities_index,10)
        recommended_item_20,_ = get_prediction_list_fata(model,i,user,target,contexts_index,entities_index,20)
        if target in recommended_item_10:
            hr1 +=1
            posi = recommended_item_10.index(target)
            ndcg1 += 1 / math.log(posi + 2,2)
        if target in recommended_item_20:
            hr2 +=1
            posi = recommended_item_20.index(target)
            ndcg2 += 1 / math.log(posi + 2,2)
        i = i + 1
    return hr1 / data.shape[0], ndcg1 / data.shape[0],hr2 / data.shape[0], ndcg2 / data.shape[0]

