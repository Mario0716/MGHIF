import os
import pickle 
import numpy as np
import json

from numpy.compat.py3k import is_pathlib_path
from src import utils
from src.metapath import config
import time
import tqdm
import torch
import torch.fft

def tostr(line):
    line = [str(l) for l in line]
    return " ".join(list(line)) + "\n"
class MetaPath(object):
    def __init__(self,graph,trainSet) -> None:
        super().__init__()
        self.graph = graph
        self.trainSet = trainSet
        self.loadPath()
        # self.dividePathset(pathsets)

    def loadPath(self):
        self.searchPath4()
        return
        if not os.path.isfile(config.ipath_filename):
            print("searching user-item path")
            self.searchPath4()
            print("saving pathset to file,",time.asctime( time.localtime(time.time()) ))
            with open(config.upath_filename,"w") as file:
                pickle.dump(self.upathset,file)
                # lines = []
                # for key in tqdm.tqdm(self.upathset.keys()):
                #     # print(map(tostr,self.upathset.get(key)))
                #     lines = lines + list(map(tostr,self.upathset.get(key)))
                # file.writelines(lines)
            with open(config.ipath_filename,"w") as file:
                pickle.dump(self.ipathset,file)
            print("saved path to file,",time.asctime( time.localtime(time.time()) ))
                # lines = []
                # for key in tqdm.tqdm(self.ipathset.keys()):
                #     lines = lines + list(map(tostr,self.ipathset.get(key)))
                # file.writelines(lines)    
                # file.write(demjson.encode({"upathset":self.upathset,"ipathset":self.ipathset}).replace("'",'"'))
        else:
            print("reading user-item path",time.asctime( time.localtime(time.time()) ))
            self.upathset = {}
            self.ipathset = {}
            s = time.time()
            with open(config.upath_filename,"r") as file:
                self.upathset = pickle.load(file)
                # lines = file.readlines()
                # for l in tqdm.tqdm(lines):
                #     l = l.split(" ")
                #     head = int(l[0])
                #     if self.upathset.get(head) is None:
                #         self.upathset[head] = [[int(node) for node in l]]
                #     else:
                #         self.upathset[head].append([int(node) for node in l])
            with open(config.ipath_filename,"r") as file:
                self.ipathset = pickle.load(file)
            print("loaded path from file",time.asctime( time.localtime(time.time()) ))
                # lines = file.readlines()
                # for l in tqdm.tqdm(lines):
                #     l = l.split(" ")
                #     head = int(l[0])
                #     if self.ipathset.get(head) is None:
                #         self.ipathset[head] = [[int(node) for node in l]]
                #     else:
                #         self.ipathset[head].append([int(node) for node in l])
        # self.searchPath4()
        # print(self.upathset)
        return 

    def searchPath4(self):
        max_path_len = config.max_path_len
        save_path_len = [4]
        threshhold = config.threshhold
        baseu2 = np.array(self.trainSet,dtype='i4')[:,:2]
        basei2 = np.array(self.trainSet,dtype='i4')[:,:2][:,[1,0]]
        # print(baseu2.shape,basei2.shape,baseu2[1,:],basei2[1,0])
        lnum = baseu2.shape[0]
        u2map = {}
        for i in range(lnum):
            head = baseu2[i][0]
            if self.trainSet[i][-1] < threshhold:
                continue
            if u2map.get(head) is None:
                u2map[head] = [baseu2[i]]
            else:
                u2map[head].append(baseu2[i])
        i2map = {}
        for i in range(lnum):
            head = basei2[i][0]
            if self.trainSet[i][-1] < threshhold:
                continue
            if i2map.get(head) is None:
                i2map[head] = [basei2[i]]
            else:
                i2map[head].append(basei2[i])
        # print(u2map)

        self.upathset = {}
        uheads = u2map.keys()
        print("Starting construct u4pathset")
        for head in tqdm.tqdm(uheads):
            fpaths = u2map.get(head)
            for fpath in fpaths:
                fhead = fpath[0]
                ftail = fpath[1]
                nheads = np.array(i2map.get(ftail),dtype='i4')[:,1]
                for nhead in nheads:
                    npaths = u2map.get(nhead)
                    for npath in npaths:
                        nhead = npath[0]
                        ntail = npath[1]
                        if nhead != fhead:
                            if self.upathset.get(fhead) is None:
                                self.upathset[fhead] = [[fhead,ftail,nhead,ntail]]
                            else:
                                self.upathset[fhead].append([fhead,ftail,nhead,ntail])
            # TODO 排序TOP K
        for head in self.upathset.keys():
            paths = np.array(self.upathset[head],dtype='i4')
            l = [i for i in range(paths.shape[0])]
            np.random.shuffle(l)
            self.upathset[head] = paths[l[:config.maxM],:]

        self.ipathset = {}
        iheads = i2map.keys()
        print("Starting construct i4pathset")
        for head in tqdm.tqdm(iheads):
            fpaths = i2map.get(head)
            for fpath in fpaths:
                fhead = fpath[0]
                ftail = fpath[1]
                nheads = np.array(u2map.get(ftail),dtype='i4')[:,1]
                for nhead in nheads:
                    npaths = i2map.get(nhead)
                    for npath in npaths:
                        nhead = npath[0]
                        ntail = npath[1]
                        if nhead != fhead:
                            if self.ipathset.get(fhead) is None:
                                self.ipathset[fhead] = [[fhead,ftail,nhead,ntail]]
                            else:
                                self.ipathset[fhead].append([fhead,ftail,nhead,ntail])
        for head in self.ipathset.keys():
            paths = np.array(self.ipathset[head],dtype='i4')
            l = [i for i in range(paths.shape[0])]
            np.random.shuffle(l)
            self.ipathset[head] = paths[l[:config.maxM],:]

    def getScoresEmb(self,users,items,score_emb_layer,uisU,iisU):
        num = min(len(users),len(items))
        scores_emb = None
        scores_idx = []
        for i in range(num):
            scores_idx.append(self.getScoreIdx(users[i],items[i],uisU,iisU))
            uisU = not uisU
            iisU = not iisU
        scores_emb = score_emb_layer(torch.LongTensor(scores_idx))
        return scores_emb
    def getScoreIdx(self,user,item,uisU,iisU):
        score_idx = 0
        if uisU != iisU:
            u = user if uisU  else item
            i = item if uisU else user
            neisi = self.graph.get(u)
            if neisi is not None :
                score = neisi.get(item)
                score_idx = 0 if score is None else score
        return score_idx

    def getPathsEmb(self,user_emb_layer,item_emb_layer,user=None,item=None):
        if user is not None:
            upaths = self.upathset.get(user) #[M1,I]
        if item is not None:
            ipaths = self.ipathset.get(item) #[M2,I]
        upaths_emb = None
        ipaths_emb = None
        # print(type(upaths),len(upaths))
        if upaths is not None and ipaths is not None:
            m1 = upaths.shape[0]
            m2 = ipaths.shape[0]
            i = upaths.shape[1]
            m = min(m1,m2,config.maxM)
            upaths = upaths[:m].reshape((m,-1,2)) #[m,i//2,2]
            ipaths = ipaths[:m].reshape((m,-1,2)) #[m,i//2,2]
            unodes = np.vstack((upaths[:,:,0],ipaths[:,:,1])).reshape((-1)) #[2,m,i//2,1]
            inodes = np.vstack((upaths[:,:,1],ipaths[:,:,0])).reshape((-1))
            # uidx = self.getNodesIdx(unodes,self.umap)
            # iidx = self.getNodesIdx(inodes,self.imap)
            uemb = user_emb_layer(torch.LongTensor(unodes)).view((2,m,i//2,-1)) #[2,m,i//2,e]
            iemb = item_emb_layer(torch.LongTensor(inodes)).view((2,m,i//2,-1))
            upaths_emb = torch.cat((uemb[0],iemb[0]),-2).view((m,i,-1))
            ipaths_emb = torch.cat((iemb[1],uemb[1]),-2).view((m,i,-1))
            upaths = upaths.reshape((m,-1))
            ipaths = ipaths.reshape((m,-1))
            
        return upaths_emb,ipaths_emb,upaths,ipaths
    def getNodesIdx(self,nodes,nmap):
        idx = []
        for node in nodes:
            idx.append(nmap.get(node))
        return torch.LongTensor(idx)

    def interaction(self, s, t):
        #s,t: B*1*E*N
        length = s.shape[-1] + t.shape[-1] - 1
        s = torch.fft.fft(s, n=length)
        t = torch.fft.fft(t, n=length)
        c = s*t
        c = torch.fft.ifft(c)
        c = c.transpose(-1,-2).float()
        return c

