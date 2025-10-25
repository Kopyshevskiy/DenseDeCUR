import torch
import torch.nn as nn
import torchvision

from .densecl import DenseCL


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class DenseDeCUR(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
    
        self.mod1 = DenseCL(pretrained=True)  
        self.mod2 = DenseCL(pretrained=True)  

        self.bn = nn.BatchNorm1d(128, affine=False)



    def bt_loss_cross_baseline(self, z1, z2):
        lambd = self.args.lambd
        dim_c = self.args.dim_common

        c = self.bn(z1).T @ self.bn(z2)

        c.div_(z1.size(0))
        torch.distributed.all_reduce(c)
  
        c_c = c[:dim_c,:dim_c]          
        c_u = c[dim_c:,dim_c:]          

        on_diag_c  = torch.diagonal(c_c).add_(-1).pow_(2).sum() 
        off_diag_c = off_diagonal(c_c).pow_(2).sum()           

        on_diag_u  = torch.diagonal(c_u).pow_(2).sum()
        off_diag_u = off_diagonal(c_u).pow_(2).sum()
        
        loss_c = on_diag_c + lambd * off_diag_c
        loss_u = on_diag_u + lambd * off_diag_u
        
        return loss_c, on_diag_c, off_diag_c, loss_u, on_diag_u, off_diag_u


    
    def bt_loss_cross(self, z1_all, z2_all):
        # z*_all: (B, D, S^2)
        assert z1_all.dim() == 3 and z2_all.dim() == 3

        dim_c = self.args.dim_common

        # matching solo spazio common
        z1_c = z1_all[:, :dim_c, :]      
        z2_c = z2_all[:, :dim_c, :]

        sim_mtx  = torch.matmul(z1_c.permute(0, 2, 1), z2_c)    # (B, S^2, S^2)
        sim_idx  = sim_mtx.max(dim=2)[1]                        # (B, S^2)

        # allinea tutte le feature (common + unique)
        z2_all = torch.gather(z2_all, 2, sim_idx.unsqueeze(1).expand(-1, z2_all.size(1), -1))

        # stack (M = B*S^2)
        Z1 = z1_all.permute(0, 2, 1).reshape(-1, z1_all.size(1)) # (M, D)
        Z2 = z2_all.permute(0, 2, 1).reshape(-1, z2_all.size(1)) # (M, D)

        return self.bt_loss_cross_baseline(Z1, Z2)



    def forward(self, y1_1,y1_2,y2_1,y2_2):

        single_loss1, dense_loss1, z1, z1_all = self.mod1(y1_1, y1_2)
        single_loss2, dense_loss2, z2, z2_all = self.mod2(y2_1, y2_2)

        loss1 = single_loss1 + dense_loss1
        loss2 = single_loss2 + dense_loss2
        
        if self.args.baseline:
            loss12_c, on_diag12_c, _, loss12_u, _, _ = self.bt_loss_cross_baseline(z1,z2)
        else:
            loss12_c, on_diag12_c, _, loss12_u, _, _ = self.bt_loss_cross(z1_all, z2_all)

        loss12 = (loss12_c + loss12_u) / 2.0

        return loss1, loss2, loss12, on_diag12_c


