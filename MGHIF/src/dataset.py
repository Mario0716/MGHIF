from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
maxun = 50
maxM = 50

class UIDataset(Dataset):
    def __init__(self,data,upathset,ipathset,uugraph) -> None:
        super().__init__()
        # data = np.array(data)
        self.data = np.array(data)
        self.upathset = upathset
        self.ipathset = ipathset
        self.uugraph = uugraph
    
    def __getitem__(self,index):
        user = self.data[index,0]
        item = self.data[index,1]
        score = self.data[index,2]

        uneis = self.uugraph.get(user)
        uneis = [user] if uneis is None else uneis
        un = len(uneis)
        uneis = uneis + [0]*(maxun-un) if un<maxun else uneis[:maxun]
        
        upaths = self.upathset.get(user)
        ipaths = self.ipathset.get(item)

        state = upaths is not None and ipaths is not None
        return state,uneis,upaths,ipaths,score
    def __len__(self):
        return self.data.shape[0]
def collate_fn(batch):
    obatch = []
    uneis_ = None
    upaths_ = None
    ipaths_ = None
    score_ = None
    for i in range(len(batch)):
        (state,uneis,upaths,ipaths,score) = batch[i]
        if state:
            if maxM > upaths.shape[0]:
                uppadding = np.array([0] * (maxM - upaths.shape[0]) * 4).reshape(-1,4)
                upaths = np.vstack((upaths,uppadding))
            if maxM > ipaths.shape[0]:
                ippadding = np.array([0] * (maxM - ipaths.shape[0]) * 4).reshape(-1,4)
                ipaths = np.vstack((ipaths,ippadding))
            uneis = torch.from_numpy(np.array(uneis)).view((-1,maxun))
            upaths = torch.from_numpy(np.array(upaths)).view((-1,maxM,4))
            ipaths = torch.from_numpy(np.array(ipaths)).view((-1,maxM,4))
            score = torch.from_numpy(np.array(score)).view((-1,1))
            uneis_ = uneis if uneis_ is None else torch.cat((uneis_,uneis),0) 
            upaths_ = upaths if upaths_ is None else torch.cat((upaths_,upaths),0) 
            ipaths_ = ipaths if ipaths_ is None else torch.cat((ipaths_,ipaths),0) 
            score_ = score if score_ is None else torch.cat((score_,score),0) 
    return uneis_,upaths_,ipaths_,score_
            


    # print(batch)
    return obatch 
    # Gather in lists, and encode labels as indices

    # elif isinstance(batch[0], float):
    #     return torch.DoubleTensor(batch)
    # elif isinstance(batch[0], string_classes):
    #     return batch

def getDataLoader(traindata,testdata,upathset,ipathset,uugraph,batch_size):
    traindst = UIDataset(traindata,upathset,ipathset,uugraph)
    testdst = UIDataset(testdata,upathset,ipathset,uugraph)
    train_loader = DataLoader(traindst,batch_size,shuffle=True,num_workers=4,collate_fn=collate_fn,pin_memory=True)
    test_loader = DataLoader(testdst,batch_size,shuffle=True,num_workers=4,collate_fn=collate_fn,pin_memory=True)
    return train_loader,test_loader
