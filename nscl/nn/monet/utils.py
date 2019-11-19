import torch
import torch.nn as nn
import jactorch.nn as jacnn

class resize_module(nn.Module):
    def __init__(self,h1,w1,h2,w2):
        super().__init__()
        self.h1, self.w1 = h1, w1
        self.h2, self.w2 = h2, w2
        self.batch_dim = batch_dim
        self.roi_pool = jacnn.PrRoIPool2D(h2, w2, 1.0)
        self._type_defined_ = False

    def forward(self,input):
        # input [batch_size, h1, w1]

        if not self._type_defined_:
            ind = torch.arange(input.shape[0],dtype=input.dtype,device=input.device).unsqueeze(-1)
            box = torch.tensor([0,0,self.h1-1,self.w1-1],dtype=input.dtype,device=input.device).unsqueeze(0).repeat(input.shape[0],1)
            self.ind_and_box = torch.cat([ind, box], dim=-1)
            self._type_defined_ = True
        return self.roi_pool(input,self.ind_and_box).squeeze() # output [batch_size, h2, w2]