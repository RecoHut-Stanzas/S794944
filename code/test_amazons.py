# -- coding:UTF-8 
import torch
# print(torch.__version__) 
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys

os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [3]))

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
 
import torch.nn.functional as F
import torch.autograd as autograd 

import pdb
from collections import defaultdict
import time
import data_utils 
import evaluate
from shutil import copyfile


dataset_base_path='../data/amazon-book'  
 
##gowalla
user_num=52643
item_num=91599 
factor_num=64
batch_size=2048*512
top_k=20
num_negative_test_val=-1##all 
 

start_i_test=150
end_i_test=200
setp=5
 

run_id="s0"
print(run_id)
dataset='amazon-book'
path_save_base='./log/'+dataset+'/newloss'+run_id
if (os.path.exists(path_save_base)):
    print('has results save path')
else:
    print('error') 
    pdb.set_trace() 
result_file=open(path_save_base+'/results_hdcg_hr.txt','a')#('./log/results_gcmc.txt','w+')
copyfile('./test_amazons.py', path_save_base+'/test_amazos'+run_id+'.py')

path_save_model_base='../newlossModel/'+dataset+'/s'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    pdb.set_trace() 

   
training_user_set,training_item_set,training_set_count = np.load(dataset_base_path+'/datanpy/training_set.npy',allow_pickle=True)
testing_user_set,testing_item_set,testing_set_count = np.load(dataset_base_path+'/datanpy/testing_set.npy',allow_pickle=True)   
user_rating_set_all = np.load(dataset_base_path+'/datanpy/user_rating_set_all.npy',allow_pickle=True).item()

def readD(set_matrix,num_):
    user_d=[] 
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i])+1)  
        user_d.append(len_set)
    return user_d
u_d=readD(training_user_set,user_num)
i_d=readD(training_item_set,item_num)
#1/(d_i+1)
d_i_train=u_d
d_j_train=i_d
#1/sqrt((d_i+1)(d_j+1)) 
# d_i_j_train=np.sqrt(u_d*i_d) 

#user-item  to user-item matrix and item-user matrix
def readTrainSparseMatrix(set_matrix,is_user):
    user_items_matrix_i=[]
    user_items_matrix_v=[] 
    if is_user:
        d_i=u_d
        d_j=i_d
    else:
        d_i=i_d
        d_j=u_d
    for i in set_matrix:
        len_set=len(set_matrix[i])  
        for j in set_matrix[i]:
            user_items_matrix_i.append([i,j])
            d_i_j=np.sqrt(d_i[i]*d_j[j])
            #1/sqrt((d_i+1)(d_j+1)) 
            user_items_matrix_v.append(d_i_j)#(1./len_set) 
    user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
    user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)

sparse_u_i=readTrainSparseMatrix(training_user_set,True)
sparse_i_u=readTrainSparseMatrix(training_item_set,False)

#user-item  to user-item matrix and item-user matrix
 

# pdb.set_trace()

class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num,user_item_matrix,item_user_matrix,d_i_train,d_j_train):
        super(BPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """     
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num) 

        for i in range(len(d_i_train)):
            d_i_train[i]=[d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i]=[d_j_train[i]]

        self.d_i_train=torch.cuda.FloatTensor(d_i_train)
        self.d_j_train=torch.cuda.FloatTensor(d_j_train)
        self.d_i_train=self.d_i_train.expand(-1,factor_num)
        self.d_j_train=self.d_j_train.expand(-1,factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)  

    def forward(self, user, item_i, item_j):    

        users_embedding=self.embed_user.weight
        items_embedding=self.embed_item.weight   
  
        gcn1_users_embedding = (torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(self.d_i_train))#*2. #+ users_embedding
        gcn1_items_embedding = (torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(self.d_j_train))#*2. #+ items_embedding
   
        gcn2_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(self.d_i_train))#*2. + users_embedding
        gcn2_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(self.d_j_train))#*2. + items_embedding
          
        gcn3_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) + gcn2_users_embedding.mul(self.d_i_train))#*2. + gcn1_users_embedding
        gcn3_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) + gcn2_items_embedding.mul(self.d_j_train))#*2. + gcn1_items_embedding
        
        gcn4_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn3_items_embedding) + gcn3_users_embedding.mul(self.d_i_train))#*2. + gcn1_users_embedding
        gcn4_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn3_users_embedding) + gcn3_items_embedding.mul(self.d_j_train))#*2. + gcn1_items_embedding
       
        gcn_users_embedding= torch.cat((users_embedding,gcn1_users_embedding,gcn2_users_embedding,gcn3_users_embedding,gcn4_users_embedding),-1)#+gcn4_users_embedding
        gcn_items_embedding= torch.cat((items_embedding,gcn1_items_embedding,gcn2_items_embedding,gcn3_items_embedding,gcn4_items_embedding),-1)#+gcn4_items_embedding#
        
        
        g0_mean=torch.mean(users_embedding)
        g0_var=torch.var(users_embedding)
        g1_mean=torch.mean(gcn1_users_embedding)
        g1_var=torch.var(gcn1_users_embedding) 
        g2_mean=torch.mean(gcn2_users_embedding)
        g2_var=torch.var(gcn2_users_embedding)
        g3_mean=torch.mean(gcn3_users_embedding)
        g3_var=torch.var(gcn3_users_embedding)
        g4_mean=torch.mean(gcn4_users_embedding)
        g4_var=torch.var(gcn4_users_embedding)
        # g5_mean=torch.mean(gcn5_users_embedding)
        # g5_var=torch.var(gcn5_users_embedding)
        g_mean=torch.mean(gcn_users_embedding)
        g_var=torch.var(gcn_users_embedding)

        i0_mean=torch.mean(items_embedding)
        i0_var=torch.var(items_embedding)
        i1_mean=torch.mean(gcn1_items_embedding)
        i1_var=torch.var(gcn1_items_embedding)
        i2_mean=torch.mean(gcn2_items_embedding)
        i2_var=torch.var(gcn2_items_embedding)
        i3_mean=torch.mean(gcn3_items_embedding)
        i3_var=torch.var(gcn3_items_embedding)
        i4_mean=torch.mean(gcn4_items_embedding)
        i4_var=torch.var(gcn4_items_embedding) 
        # i5_mean=torch.mean(gcn5_items_embedding)
        # i5_var=torch.var(gcn5_items_embedding)
        i_mean=torch.mean(gcn_items_embedding)
        i_var=torch.var(gcn_items_embedding)

        # pdb.set_trace() 

        str_user=str(round(g0_mean.item(),7))+' '
        str_user+=str(round(g0_var.item(),7))+' '
        str_user+=str(round(g1_mean.item(),7))+' '
        str_user+=str(round(g1_var.item(),7))+' '
        str_user+=str(round(g2_mean.item(),7))+' '
        str_user+=str(round(g2_var.item(),7))+' '
        str_user+=str(round(g3_mean.item(),7))+' '
        str_user+=str(round(g3_var.item(),7))+' '
        str_user+=str(round(g4_mean.item(),7))+' '
        str_user+=str(round(g4_var.item(),7))+' '
        # str_user+=str(round(g5_mean.item(),7))+' '
        # str_user+=str(round(g5_var.item(),7))+' '
        str_user+=str(round(g_mean.item(),7))+' '
        str_user+=str(round(g_var.item(),7))+' '

        str_item=str(round(i0_mean.item(),7))+' '
        str_item+=str(round(i0_var.item(),7))+' '
        str_item+=str(round(i1_mean.item(),7))+' '
        str_item+=str(round(i1_var.item(),7))+' '
        str_item+=str(round(i2_mean.item(),7))+' '
        str_item+=str(round(i2_var.item(),7))+' '
        str_item+=str(round(i3_mean.item(),7))+' '
        str_item+=str(round(i3_var.item(),7))+' '
        str_item+=str(round(i4_mean.item(),7))+' '
        str_item+=str(round(i4_var.item(),7))+' '
        # str_item+=str(round(i5_mean.item(),7))+' '
        # str_item+=str(round(i5_var.item(),7))+' '
        str_item+=str(round(i_mean.item(),7))+' '
        str_item+=str(round(i_var.item(),7))+' '

        print(str_user)
        print(str_item)


        return gcn_users_embedding, gcn_items_embedding,str_user,str_item 


 
 
test_batch=52#int(batch_size/32) 
testing_dataset = data_utils.resData(train_dict=testing_user_set, batch_size=test_batch,num_item=item_num,all_pos=training_user_set)
testing_loader = DataLoader(testing_dataset,batch_size=1, shuffle=False, num_workers=0) 
 

model = BPR(user_num, item_num, factor_num,sparse_u_i,sparse_i_u,d_i_train,d_j_train)
model=model.to('cuda')
   
optimizer_bpr = torch.optim.Adam(model.parameters(), lr=0.001)#, betas=(0.5, 0.99))

########################### TRAINING ##################################### 
# testing_loader_loss.dataset.ng_sample() 

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

print('--------test processing-------')
count, best_hr = 0, 0
for epoch in range(start_i_test,end_i_test,setp):
    model.train()   

    PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
    #torch.save(model.state_dict(), PATH_model) 
    model.load_state_dict(torch.load(PATH_model)) 
    model.eval()     
    # ######test and val###########    
    gcn_users_embedding, gcn_items_embedding,gcn_user_emb,gcn_item_emb= model(torch.cuda.LongTensor([0]), torch.cuda.LongTensor([0]), torch.cuda.LongTensor([0])) 
    user_e=gcn_users_embedding.cpu().detach().numpy()
    item_e=gcn_items_embedding.cpu().detach().numpy()
    all_pre=np.matmul(user_e,item_e.T) 
    HR, NDCG = [], [] 
    set_all=set(range(item_num))  
    #spend 461s 
    test_start_time = time.time()
    for u_i in testing_user_set: 
        item_i_list = list(testing_user_set[u_i])
        index_end_i=len(item_i_list)
        item_j_list = list(set_all-training_user_set[u_i]-testing_user_set[u_i])
        item_i_list.extend(item_j_list) 

        pre_one=all_pre[u_i][item_i_list] 
        indices=largest_indices(pre_one, top_k)
        indices=list(indices[0])   

        hr_t,ndcg_t=evaluate.hr_ndcg(indices,index_end_i,top_k) 
        elapsed_time = time.time() - test_start_time 
        HR.append(hr_t)
        NDCG.append(ndcg_t)    
    hr_test=round(np.mean(HR),4)
    ndcg_test=round(np.mean(NDCG),4)    
        
    # test_loss,hr_test,ndcg_test = evaluate.metrics(model,testing_loader,top_k,num_negative_test_val,batch_size)  
    str_print_evl="epoch:"+str(epoch)+'time:'+str(round(elapsed_time,2))+"\t test"+" hit:"+str(hr_test)+' ndcg:'+str(ndcg_test) 
    print(str_print_evl)   
    result_file.write(gcn_user_emb)
    result_file.write('\n')
    result_file.write(gcn_item_emb)
    result_file.write('\n') 

    result_file.write(str_print_evl)
    result_file.write('\n')
    result_file.flush()

 

 
#788s 
    # test_start_time = time.time()
    # for u_i in testing_user_set:#这里的self.data_set_count=1=batch_size 
    #     item_i_list = list(testing_user_set[u_i])
    #     index_end_i=len(item_i_list)
    #     item_j_list = list(set_all-training_user_set[u_i]-testing_user_set[u_i])
    #     item_i_list.extend(item_j_list)
    #     pre_one=all_pre[u_i][item_i_list]
    #     indices=np.argsort(pre_one)[-top_k:][::-1]
    #     # pdb.set_trace()
    #     # _, indices = torch.topk(pre_one, top_k)   
    #     hr_t,ndcg_t=evaluate.hr_ndcg(indices,index_end_i,top_k) 
    #     elapsed_time = time.time() - test_start_time
    #     print('time:'+str(round(elapsed_time,2)))  
    #     HR.append(hr_t)
    #     NDCG.append(ndcg_t)  
        
    #     # pdb.set_trace()
    # hr_test=round(np.mean(HR),4)
    # ndcg_test=round(np.mean(NDCG),4)   

 


