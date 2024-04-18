import torch
import torch.nn as nn

import numpy as np

from lammps import lammps
from os.path import join
from os import listdir
import re
from ase.io import read,write

from functools import partial

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

"""
Hack from https://github.com/pytorch/pytorch/issues/91810 to acess storage of grad-tracking tensors
"""
def detach_numpy(tensor):
    tensor = tensor.detach().cpu()
    if torch._C._functorch.is_gradtrackingtensor(tensor):
        tensor = torch._C._functorch.get_unwrapped(tensor)
        return np.array(tensor.storage().tolist()).reshape(tensor.shape)
    return tensor.numpy()

class SNAPDescriptor(torch.autograd.Function):
    @staticmethod
    def forward(X):
        _LMP.set_X(detach_numpy(X))
        return torch.tensor(_LMP.get_D()) # for backward pass

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        X, = inputs
        ctx.save_for_backward(X.detach())

    @staticmethod
    def backward(ctx,grad_f_d):
        X,=ctx.saved_tensors
        _LMP.set_X(detach_numpy(X))
        return torch.einsum("ijk,k->ij",torch.tensor(_LMP.get_dD()),grad_f_d)

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

        return torch.tensor(conf_ini.positions),torch.tensor(conf_fin.positions)
    

data_set = PtNanoparticleDataset(data_path='data/transition_data_pt/')
data_loader = DataLoader(data_set,shuffle=True)


global_steps = 100
local_euler_steps = 100

eta_euler = 5e-6
eta_metric = 1e-3
grad_tol = 1e-1

def energy(x,a,D_f):
    return 0.5 * torch.sum(((D_SNAP(x)-D_f) @ a)**2)


rk = 20
A = torch.randn(55,rk,dtype=torch.double)/np.sqrt(rk*55) # A@A.T parametrizes low-rank pseudometric

def reconstruction_loss(x,y):
    return torch.mean((x-y)**2)

grad_x_reconstruction_loss = torch.func.jacrev(reconstruction_loss,0) # (N,3)
grad_x_energy = torch.func.grad(energy,0) # (N,3)

ini_loss_hist = []
fin_loss_hist = []

# print(torch.autograd.functional.jacobian(f,A).shape) # (N,3,N_d,rk)
# jac_d = torch.func.jacrev(D_SNAP) # (N_D x N x 3)


for step in range(global_steps):
    X,X_f = next(iter(data_loader))
    X = X.squeeze(0)
    X_f = X_f.squeeze(0)
    D_f = D_SNAP(X_f)

    l_ini = reconstruction_loss(X,X_f)
    print(f"Global iteration {step}, initial reconstruction loss: {l_ini}")

    ini_loss_hist.append(l_ini)
    k = 0


    while True:
        e = energy(X,A,D_f)
        g = grad_x_energy(X,A,D_f)
        g_linf = torch.abs(g).max()
        if g_linf < grad_tol:
            break
        X -= eta_euler * g
        k +=1 
        if k%50 == 0:
            print(f"\tEuler iteration {k}, loss: {e}, max abs gradient {g_linf}")
    
    J = torch.autograd.functional.jacobian(lambda a: grad_x_energy(X,a,D_f),A) # (N,3,N_d,rk), probably a much more efficient method..

    l_fin = reconstruction_loss(X,X_f)
    print(f"Global iteration {step}, initial reconstruction loss: {l_fin}")
    fin_loss_hist.append(l_fin)

    A -= eta_metric * torch.einsum('ij,ijkl->kl',J,grad_x_reconstruction_loss(X,X_f))
    torch.save(A,"weights_reconstruct.pt")