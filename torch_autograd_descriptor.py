import torch
import torch.nn as nn

import numpy as np

from lammps import lammps
from os.path import join
from os import listdir
import re
from ase.io import read,write

from torch.utils.data import Dataset,DataLoader


data_dir = "data/transition_data_pt/"

def init_commands(s1,s2,i):
    return f"""
    units metal
    boundary p p p
    atom_modify map yes
    
    read_data {join(data_dir,f"{s1}_{s2}_{i}.dat")}
    mass * 195.1

    pair_style snap
    pair_coeff * * W.snapcoeff W.snapparam W

    fix seal all nve

    compute D all sna/atom 4.7 0.99363 8 0.5 1
    compute dD all snad/atom 4.7 0.99363 8 0.5 1

    run 0
"""

class LammpsComputeSNAP():
    def __init__(self,s1=1,s2=2,N_D=55):
        
        self.N_D = N_D
        
        self.L = lammps(cmdargs=['-log','none','-screen','none'])
        self.L.commands_string(init_commands(s1,s2,0))

        self.D_target = self.get_D() # N x N_D
        self.X_target = self.get_X() # N x 3

        self.N = self.X_target.shape[0]

    def get_D(self):
        return np.ctypeslib.as_array(
            self.L.gather("c_D",1,self.N_D)).reshape((-1,self.N_D)).sum(0) # gradients are wrt descriptor summed over atoms

    def get_dD(self):
        """ snad/atom returns negative gradient! -- w.r.t. descriptor summed over atoms"""
        return -np.ctypeslib.as_array(
            self.L.gather("c_dD",1,3*self.N_D)).reshape((-1,3,self.N_D))

    def get_X(self):
        return np.ctypeslib.as_array(
            self.L.gather("x",1,3)).reshape((-1,3))

    def set_X(self,X):
        self.L.scatter("x",1,3,np.ctypeslib.as_ctypes(X.flatten()))
        self.L.command("run 0")
    
_LMP = LammpsComputeSNAP()

class SNAPDescriptor(torch.autograd.Function):
    @staticmethod
    def forward(ctx,X):
        ctx.save_for_backward(X)
        _LMP.set_X(X.numpy())
        return torch.tensor(_LMP.get_D())

    @staticmethod
    def backward(ctx,grad_loss_D):
        X, = ctx.saved_tensors
        _LMP.set_X(X.numpy())
        grad_D_X = torch.tensor(_LMP.get_dD())
        return torch.einsum("ijk,k->ij",grad_D_X,grad_loss_D)

D_SNAP = SNAPDescriptor.apply

def read_ase(filename):
    return read(
        filename=filename, 
        format="lammps-data", 
        Z_of_type={1: 78}, 
        style="atomic")

class PtNanoparticleDataset(Dataset):
    def __init__(self,data_path):
        self.data_path = data_path
        self.files = listdir(self.data_path)
        self.transitions = []

        for f in self.files:
            m = re.match('(\d+)_(\d+)_([01])\.dat',f)
            if m is not None:
                state_ini, state_fin, is_final = map(int, m.groups())
                if is_final == 0:
                    self.transitions.append((state_ini,state_fin,f,f"{state_ini}_{state_fin}_1.dat"))
    
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self,idx):
        s1,s2,f1,f2 = self.transitions[idx]
        conf_ini = read_ase(join(self.data_path,f1))
        conf_fin = read_ase(join(self.data_path,f2))

        return torch.tensor(conf_ini.positions,requires_grad = True),torch.tensor(conf_fin.positions)
    

data_set = PtNanoparticleDataset(data_path='data/transition_data_pt/')
data_loader = DataLoader(data_set,shuffle=True)


global_steps = 100
local_euler_steps = 100

eta_euler = 1e-6
eta_metric = 1e-6

A = torch.randn(55,10,requires_grad=True,dtype=torch.double)/55 # A@A.T parametrizes low-rank pseudometric
A.retain_grad()

class MetricModel(nn.Module):
    def __init__(self,rank,dim_features):
        super().__init__()
        self.A = torch.randn(dim_features,rank,dtype=torch.double) / torch.sqrt(dim_features*rank)

for step in range(global_steps):
    A.grad = None # reset A grad
    X,X_f = next(iter(data_loader))

    X = X.squeeze(0)
    X_f = X_f.squeeze(0)

    print(f"Step: {step}, initial distance: {torch.mean((X-X_f)**2)}")
    D_f = D_SNAP(X_f)

    gloss_hist = []
    traj_hist = []

    rk = 10


    for k in range(100):
        traj_hist.append(X.detach().numpy())
        X.grad = None
        X.retain_grad() # X is not a leaf node
        D = D_SNAP(X)
        loss = torch.mean(((D-D_f)@A)**2)
        loss.backward(retain_graph=True)
        if k%10 == 0:
            print(f"Euler step {k}, loss {loss.item()}")

        with torch.no_grad():
            X -= eta_euler*X.grad

    glob_loss = torch.mean((X-X_f)**2) ## really here, we should be looking at RMSD
    glob_loss.backward()
    gloss_hist.append(glob_loss.item())
    print(f"Step: {step}, final distance: {gloss_hist[-1]}")

    with torch.no_grad():
        A -= eta_metric*A.grad
