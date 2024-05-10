from src.fc import FCNet
import torch
import torch.nn as nn

class uiAttention(nn.Module):
    def __init__(self,dim) -> None:
        super().__init__()
        self.dim = dim
        self.lR = FCNet([2 * dim, dim], 'ReLU', 0, bias = True)
        self.lout = FCNet([dim, dim], 'ReLU', 0, bias = True)

        self.l1 = FCNet([3*dim, dim], 'ReLU', 0, bias = True)
        self.l2 = FCNet([dim, 1], '', 0, bias = True)

    def forward(self,user,item,pus,pis,pss,C):
        """
        pus : users in path [M,N,E]
        pis : items in path [M,N,E]
        pss : scores in path [M,N,E]
        C : cross fusion representation of user and item [M,N,E] 
        return pair_emb [M,E]
        """
        M = C.size(-3)
        w = torch.softmax(self.l2(self.l1(torch.cat((pus,pis,pss),-1))),-2).transpose(-1,-2) # [M,1,N]
        pair_emb = torch.matmul(w,self.lR(torch.cat((C,pss),-1)))
        pair_emb = self.lout(pair_emb).view((M,self.dim)) #[M,1,E]
        return pair_emb

class uiPAttention(nn.Module):
    def __init__(self,dim) -> None:
        super().__init__()
        self.dim = dim
        self.lR = FCNet([dim, dim], 'ReLU', 0, bias = True)
        self.lout = FCNet([dim, dim], 'ReLU', 0, bias = True)

        self.l1 = FCNet([dim, dim], 'ReLU', 0, bias = True)
        self.l2 = FCNet([dim, 1], '', 0, bias = True)
    def forward(self,C):
        # B = C.size(0)
        M = C.size(-3)
        w = torch.softmax(self.l2(self.l1(C)),-2).transpose(-1,-2) # [B,M,1,N]
        pair_emb = torch.matmul(w,self.lR(C))
        pair_emb = self.lout(pair_emb).view((M,self.dim)) #[b,M,E]
        return pair_emb


class uuAttention(nn.Module):
    def __init__(self,dim) -> None:
        super().__init__()
        self.lout = FCNet([dim, dim], 'ReLU', 0, bias = True)

        self.l1 = FCNet([2*dim, dim], 'ReLU', 0, bias = True)
        self.l2 = FCNet([dim, 1], '', 0, bias = True)
    def forward(self,user,neis):
        """
        user [B,E]
        neis [B,U,E]
        return uout: [B,1,E]
        """
        U = neis.size(-2)
        # B = neis.size(0)
        user = user.view(1,-1).repeat((U,1))
        # user = user.unsqueeze(1).repeat((1,U,1))
        w = torch.softmax(self.l2(self.l1(torch.cat((user,neis),-1))),-2).transpose(-1,-2) # [B,1,U]
        user_ = torch.matmul(w,neis)
        uout = self.lout(user_)
       
        return uout


