from os import error
from pickle import NONE

from numpy.lib.utils import source
from src.dataset import getDataLoader
from src.metapath.metapath import MetaPath
from src.metapath.attention import uiAttention,uuAttention,uiPAttention
from src.metapath import config
from src.measure import Measure
from src import utils
from src.BAN.fusion import BAN
from torch.autograd import Variable
import tqdm
import time
import torch
import torch.nn as nn
import numpy as np
class MGHIF(nn.Module):
    def __init__(self):
        super().__init__()
        self.setup_seed(config.seed)
        print("loading user-item rating data")
        self.graph, self.trainData, self.testData,self.umap, uidx,iidx,sidx = utils.read_rating_edges(config.train_filename,config.test_filename)
        self.uugraph,uidx = utils.read_uu_edges(config.uu_filename,"",self.umap,uidx)
        self.mp = MetaPath(self.graph,self.trainData)

        print("TrainSet edges: ",len(self.trainData))
        print("TestSet edges: ",len(self.testData))
        self.emb_dim = config.emb_dim

        self.user_emb_layer = nn.Embedding(uidx, self.emb_dim, padding_idx=0)
        self.item_emb_layer = nn.Embedding(iidx, self.emb_dim, padding_idx=0)
        self.score_emb_layer = nn.Embedding(sidx, self.emb_dim, padding_idx=0)
        self.load_preuseremb()
        self.uiAtt = uiAttention(config.emb_dim)
        self.uuAtt = uuAttention(config.emb_dim)
        self.dnn = BAN(self.emb_dim,self.emb_dim,1)
        self.final_linear = nn.Sequential(
            nn.Linear(3*self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.emb_dim, 1)
        )
        self.measure = Measure()
        # self.mp.getC("u11","i3",self.user_emb_layer,self.item_emb_layer)
    def setup_seed(self,seed):
        if seed is None:
          seed = np.random.randint(10000)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # random.seed(seed)
        torch.backends.cudnn.deterministic = True
        print("Random seed:",seed)
    def load_preuseremb(self):
        neis_count = {}
        with open(config.uu_filename,"r") as f:
          lines = f.readlines()
          for line in lines:
            l = line.split(" ")
            s = int(l[0])
            t = int(l[0])
            r = int(l[0])
            if neis_count.get(s) is None:
              neis_count[s] = 1 # [t]
            else:
              neis_count[s] = neis_count[s] + 1
            if neis_count.get(t) is None:
              neis_count[t] = 1 # [s]
            else:
              neis_count[t] = neis_count[t] + 1
        valid = []
        for key in neis_count.keys():
          if neis_count.get(key)>400:
            valid.append(key)
        print("valid user: ",len(valid),'of',len(neis_count))
        W = self.user_emb_layer.weight.data.clone()
        with open(config.embfile, "r") as f:
            lines = f.readlines()[1:]  # skip the first line
            for line in lines:
                emd = line.split()
                if int(emd[0]) in valid:
                  idx = self.umap.get(int(emd[0]))
                  w = torch.tensor(utils.str_list_to_float(emd[1:]))
                  assert w.size(-1) == config.emb_dim
                  W[idx] = w
        self.user_emb_layer.weight = torch.nn.Parameter(W)
        print("pretrained user emb is loaded")
        pass
    def iter_train(self,user,item):
        # print("one pred:",user,item)
        s = time.time()
        source_feature = self.user_emb_layer(torch.LongTensor([user]))
        target_feature = self.item_emb_layer(torch.LongTensor([item]))
        neis_emb = self.getUserNeisEmb(user)

        source_emb = self.uuAtt(source_feature,neis_emb if neis_emb is not None else source_feature) # [1,E]
        upaths_emb,ipaths_emb,upaths,ipaths = self.mp.getPathsEmb(self.user_emb_layer,self.item_emb_layer,user=user,item=item)
        # [M1,I,E] [M1,I,E]
        if upaths_emb is not None and ipaths_emb is not None:
            cross_fusion = self.mp.interaction(upaths_emb.transpose(-1,-2),ipaths_emb.transpose(-1,-2))
            # [M1,2I-1,E]
            # """
            I = upaths_emb.size(1)
            M = cross_fusion.size(0)
            N = cross_fusion.size(1)
            pus = torch.zeros_like(cross_fusion)
            pis = torch.zeros_like(cross_fusion)
            pss = torch.zeros_like(cross_fusion)
            for i in range(M):
                for j in range(N):
                    isfront = j<= N//2
                    ustart = 0 if isfront else j-N//2 
                    uend = j+1 if isfront else N//2+1
                    istart = N//2-j if isfront else 0
                    iend = N//2+1 if isfront else N-j
                    pus[i,j] = upaths_emb[i,ustart:uend].sum(-2)
                    pis[i,j] = ipaths_emb[i,istart:iend].sum(-2)
                    semb = self.mp.getScoresEmb(upaths[i,ustart:uend],ipaths[i,istart:iend],self.score_emb_layer,uisU=ustart%2==0,iisU=istart%2==1)
                    if semb is not None:
                        pss[i,j] = semb.sum(-2)
            pair_emb = self.uiAtt(user,item,pus,pis,pss,cross_fusion) # [M,E]
        else:
            pair_emb = source_feature * target_feature # [1,E]
        pred,att = self.dnn(pair_emb.unsqueeze(0),source_emb.unsqueeze(0))
        pred = torch.cat((source_feature,target_feature,pred),1) #3E 
        pred = self.final_linear(pred)          
        pred = torch.sigmoid(pred) * 5
        if torch.isnan(pred):
            print(source_feature,target_feature)
            print(neis_emb,pair_emb)
        return pred
    def getUserNeisEmb(self,user):
        neis = self.uugraph.get(user)
        return self.user_emb_layer(torch.LongTensor(neis)) if neis is not None else None
    def pred_batch(self,uneis,upaths,ipaths):
        # [B,M,4]
        B = uneis.size(0)
        user = upaths[:,0,0].squeeze()
        item = ipaths[:,0,0].squeeze()
        source_feature = self.user_emb_layer(user).view((B,-1)) # [B,E]
        target_feature = self.item_emb_layer(item).view((B,-1)) # [B,E]
        un_emb = self.user_emb_layer(uneis) # [B,U,E]
        up_emb = torch.stack((self.user_emb_layer(upaths[:,:,0]),self.item_emb_layer(upaths[:,:,1]),self.user_emb_layer(upaths[:,:,2]),self.item_emb_layer(upaths[:,:,3])),3) # [B,M,E,4]
        ip_emb = torch.stack((self.item_emb_layer(ipaths[:,:,0]),self.user_emb_layer(ipaths[:,:,1]),self.item_emb_layer(ipaths[:,:,2]),self.user_emb_layer(ipaths[:,:,3])),3)
        source_emb = self.uuAtt(source_feature,un_emb) #[B,E]
        cross_fusion = self.mp.interaction(up_emb,ip_emb) #[B,M,7,E]
        pair_emb = self.uiAtt(cross_fusion) # [B,M,E]
        pred,att = self.dnn(pair_emb,source_emb)
        pred = torch.cat((source_feature,target_feature,pred.view(B,-1)),-1) #3E 
        pred = self.final_linear(pred)    
        return pred

    def train(self):
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad,
                                    self.parameters()),
                            lr=config.lr_default, betas=(0.9, 0.999), eps=1e-8) 
        for epoch in range(config.epochs):
            print("Starting Epoch ",epoch)
            Tloss = 0
            epochStart = True
            pbar = tqdm(total=len(self.train_loader))
            for i,(uneis,upaths,ipaths,score) in enumerate(self.train_loader):
                if upaths is None:
                    continue
                uneis = Variable(uneis).long()
                upaths = Variable(upaths).long()
                ipaths = Variable(ipaths).long()
                score = Variable(score).float()
                pred = self.pred_batch(uneis,upaths,ipaths)
                loss = torch.nn.functional.mse_loss(pred,score)
                loss.backward()
                optim.step()
                Tloss = Tloss + loss.data.item()        
                pbar.update(1)
            print("Epoch",epoch,"training total Loss:",Tloss)
            result = self.eval(epoch)
            print(result[0],result[1])
            with open(config.result_file,"a") as f:
                f.writelines(["Epoch " + str(epoch)+"\n"]+result)

    def train1(self):
        """
        users: [B,1] str users id
        items: [B,1] str items id
        u_emb: [B,1,E] user embedding
        i_emb: [B,1,E] item embedding
        u_neis_emb: [B,U,E] user's neis embedding in user-user network
        pus : users in path [B,M,N,<K>,E]
        pis : items in path [B,M,N,<K>,E]
        pss : scores in path [B,M,N,<K>,E]
        C : cross fusion representation of user and item [B,M,N,E] 
        """
        train_size = len(self.trainData)
        start_list = list(range(0,train_size,config.batch_size))
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad,
                                    self.parameters()),
                            lr=config.lr_default, betas=(0.9, 0.999), eps=1e-8) 
        lossItem = []
        for epoch in range(config.epochs):
            print("Starting Epoch ",epoch)
            np.random.shuffle(start_list)
            Loss = 0
            epochStart = True
            for start in tqdm.tqdm(start_list):
                end = start + config.batch_size 
                batchData = self.trainData[start:end]
                if epochStart:
                    optim.zero_grad()
                    if epoch > 1 and lossItem[-1]>lossItem[-2]:
                      for i in range(len(optim.param_groups)):
                        optim.param_groups[i]['lr'] *= config.lr_decay_rate
                    print('lr: %.6f' % optim.param_groups[-1]['lr'])
                    epochStart = False
                scores = torch.from_numpy(np.array(batchData)[:,2]).to(torch.float64)
                if scores.shape[0] == 0:
                    continue
                pred = None 
                s = time.time()
                for user,item,score in batchData:
                    onepred = self.iter_train(user,item)
                    pred = onepred if pred is None else torch.cat((pred,onepred),0)
                pred = pred.squeeze().to(torch.float64)
                loss = torch.nn.functional.mse_loss(pred,scores)     
                try:            
                    loss.backward()
                except error:
                    print(error)
                optim.step()
                Loss = Loss + loss.data.item()
            print("Epoch",epoch,"training total Loss:",Loss)
            lossItem.append(Loss)
            result = self.eval(epoch)
            print(result[0],result[1])
            with open(config.result_file,"a") as f:
                f.writelines(["Epoch " + str(epoch)+"\n"]+result)
        return
    def eval(self,epoch):
        test_size = len(self.testData)
        start_list = list(range(0,test_size,config.batch_size))
        np.random.shuffle(start_list)
        res = []
        for start in tqdm.tqdm(start_list):
            end = start + config.batch_size 
            batchData = self.testData[start:end] #[B,3]
            pred = None 
            for user,item,score in batchData:
                onepred = self.iter_train(user,item)
                pred = onepred if pred is None else torch.cat((pred,onepred),0)
            if pred == None:
                continue
            pred = pred.to(torch.float32)
            bres = np.hstack((np.array(batchData),pred.detach().numpy()))
            res = res + bres.tolist()
        eval_result = self.measure.ratingMeasure(res)
        return eval_result
            



if __name__ == "__main__": 
    modle = MGHIF()
    modle.train1()

