
import torch
import torch.nn as nn
import torch.nn.functional as F

import scnn.utils as utils

from math import ceil, cos, sin, atan2, pi
import numpy as np

# weight_init = [ [ 0, 0],
#     [-1, 1], [-2, 2],
#     [-1, 0], [-2, 0],
#     [-1,-1], [-2,-2],
#     [ 0,-1], [ 0,-2],
#     [ 1,-1], [ 2,-2],
#     [ 1, 0], [ 2, 0],
#     [ 1, 1], [ 2, 2],
#     [ 0, 1], [ 0, 2] ]
# weight_init = 1.0*torch.tensor(weight_init).view(-1,2,1)



seed = 1
np.random.seed(seed)

weight_init = [ [[1.5,0],[1,0]],
                [[1,0],[1.5,0]],
                [[2.5,0],[2,0]],
                [[1,0],[1,0.5]],
                [[1,0],[1,-0.5]],
                [[1,1],[1,0.5]],
                [[1,-1,],[1,-0.5]],
                [[1,1],[0.5,1]],
                [[1,-1],[0.5,-1]],
                [[-0.5,1],[0,1]],
                [[-0.5,-1],[0,-1]],
                [[-1.5,1],[-1,1]],
                [[-1.5,-1],[-1,-1]] ]
weight_init = 1.0*torch.tensor(weight_init).float().transpose(1,2)

''' ... '''

xy = [-5,-3,-1.8,-1,-0.6,-0.4,0,0.4,0.6,1,1.8,3,5]
xy = [-3,-1.8,-1,-0.6,-0.4,0,0.4,0.6,1,1.8,3]
xy = [-2,-.8,0,.8,2]
xy = [-2.5,-2.0,-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0,2.5] # uniform
xy = [-2.0, -1, 0, 1, 2] # uniform
xy = [-5.5, -2.75, 0, 2.75, 5.5] # uniform
loc = np.array(np.meshgrid(xy, xy)).T.reshape(-1,2)

n_tht = 4
vec = np.zeros([n_tht,2,2])
for i in range(n_tht):
    tht = np.pi*2*i/n_tht
    vec[i,:,0] = np.sin(tht), np.cos(tht)
vec[:,:,1] = -vec[:,:,0]
vec *= 1/4

weight_init = np.zeros([len(loc)*n_tht,2,2])
for i in range(len(loc)):
    weight_init[i*n_tht:(i+1)*n_tht,0,:] = loc[i][0]+vec[:,0,:]+np.random.random([4,1])
    weight_init[i*n_tht:(i+1)*n_tht,1,:] = loc[i][1]+vec[:,1,:]+np.random.random([4,1])

weight_init = torch.tensor(weight_init) + torch.normal(0,.05,size=weight_init.shape)

''' random generation '''
distance_bound = 6
n_pattern = 100
len_min, len_max = 0.3, 0.9 # meter per 0.4sec

weight_init = np.empty([n_pattern,2,2])
import scipy.stats
sigma = 4
weight_init[:,:,0] = scipy.stats.truncnorm.rvs( -distance_bound/sigma, distance_bound/sigma, loc=0, scale=sigma, size=[n_pattern,2] )
# weight_init[:,:,0] = np.random.uniform(-distance_bound,distance_bound,size=[n_pattern,2])
weight_init[:,:,1] = weight_init[:,:,0]

scaler = np.empty([n_pattern,2])
np.random.seed(seed)
scaler[:,0] = np.random.uniform(len_min,len_max,size=n_pattern)
scaler[:,1] = scaler[:,0]

np.random.seed(seed)
tht = np.random.random(n_pattern)*np.pi*2
vec = np.empty(weight_init.shape)
vec[:,:,0] = np.array([np.sin(tht), np.cos(tht)]).T *scaler /2
vec[:,:,1] = -vec[:,:,0]

weight_init = torch.tensor( weight_init + vec )


''' ... '''

def get_fc_size(ag,in_size):
    if ag.n_fc<0:
        return [in_size,*ag.fc_width,ag.output_size]
    factor = (in_size/out_size)**(1/(ag.n_fc+1))
    fc_size = [in_size]
    for i in range(ag.n_fc):
        fc_size.append(1+int(fc_size[-1]/factor))
    fc_size.append(ag.output_size)
    return fc_size


class LocPredictor(nn.Module):
    def __init__(self, ag):

        super().__init__()
        self.ag = ag
        Conv_ = L2Dist1d if ag.l2d else nn.Conv1d
        self.actv = torch.tanh if ag.tanh else F.leaky_relu
        self.p = self.ag.drop_rate

        t_conv, c_conv = [], []
        if ag.cont_ker_num[0]<0: ag.cont_ker_num[0] = len(weight_init)
        t_n_ch, c_n_ch = [ag.n_ch, *ag.targ_ker_num], [ag.n_ch, *ag.cont_ker_num]
        targ_len, cont_len = ag.hist_len, ag.hist_len
        for i in range(len(t_n_ch)-1):
            Conv = Conv_ if i==0 else nn.Conv1d
            t_conv.append( Conv(t_n_ch[i],t_n_ch[i+1],ag.targ_ker_size[i]) )
            targ_len = ceil((targ_len-ag.targ_ker_size[i]+1)/ag.targ_pool_size[i])
        for i in range(len(c_n_ch)-1):
            Conv = Conv_ if i==0 else nn.Conv1d
            c_conv.append( Conv(c_n_ch[i],c_n_ch[i+1],ag.cont_ker_size[i]) )
            cont_len = ceil((cont_len-ag.cont_ker_size[i]+1)/ag.cont_pool_size[i])

        if self.ag.neighbor:
            self.mlp_inp_size = int( targ_len*t_n_ch[-1] + cont_len*c_n_ch[-1] )
        else:
            self.mlp_inp_size = int( targ_len*t_n_ch[-1])

        fc = []
        fc_size = get_fc_size(ag, self.mlp_inp_size)
        for i in range(len(fc_size)-1):
            fc.append(nn.Linear(fc_size[i],fc_size[i+1]))

        self.t_conv, self.c_conv = nn.ModuleList(t_conv), nn.ModuleList(c_conv)
        self.fc = nn.ModuleList(fc)

        if self.ag.lock_l2d:
            self.lock_weights()
        print(self)

    def encodeTraj(self, th, ch):
        for i in range(len(self.t_conv)):
            th = F.max_pool1d( self.actv(self.t_conv[i](th)) , self.ag.targ_pool_size[i] ,ceil_mode=True)
        th = th.view(len(th), -1)
        for i in range(len(self.c_conv)):
            ch = F.max_pool1d( self.actv(self.c_conv[i](ch)) , self.ag.cont_pool_size[i] ,ceil_mode=True).float()
        ch = ch.view(len(ch), -1)
        ch = F.softmax(ch) # softmax enforces probablity distribution property
        return (th,ch)


    def forward(self, th, ch, ei): # th, ch, ei :  targ_hist, cont_hist,
        th,ch = self.encodeTraj(th, ch)
        x = []
        if self.ag.neighbor:
            for i in range(len(th)):
                if self.ag.use_max or len(ch[ei[i]:ei[i+1]])==0:
                    x.append( torch.cat([th[i],ch[ei[i]:ei[i+1]].sum(0)]) )
                else:
                    x.append( torch.cat([th[i],ch[ei[i]:ei[i+1]].max(0).values]) )
            x = torch.stack(x)
        else:
            x = th
        for i in range(len(self.fc)-1):
            x = F.dropout(F.leaky_relu(self.fc[i](x)),p=self.p)
        x = self.fc[-1](x)
        return x


    def predictTraj(self,hist,ei=None):
        if ei is None: ei = [0,len(hist)]
        n_traj, n_scene, hist_len, fut_len = len(hist), len(ei)-1, self.ag.hist_len, self.ag.fut_len
        # fut_len = 1
        _d = hist.device
        Bias, Rot = torch.Tensor(n_traj,2).to(_d), torch.Tensor(n_traj,2,2).to(_d)
        pred = torch.Tensor(n_traj,2,fut_len).to(_d)
        for t in range(fut_len):
            targ_hist, cont_hist, cont_len, i_traj = [], [], [], -1
            for i in range(n_scene):
                data_scene = hist[ei[i]:ei[i+1]]
                n_ped = len(data_scene)
                cont_len = np.append(cont_len,[n_ped-1]*n_ped)
                _idx = np.arange(n_ped)
                for j in range(n_ped):
                    i_traj+=1
                    Bias[i_traj] = data_scene[j,:,-1]
                    dt = data_scene - Bias[i_traj].view(-1,1)
                    tht = pi/2+atan2(*dt[j,:,-2])
                    Rot[i_traj] = torch.tensor([[cos(tht),-sin(tht)],[sin(tht),cos(tht)]])
                    dt = (Rot[i_traj]@dt.transpose(0,1).reshape(2,-1)).view(2,-1,hist_len).transpose(0,1)
                    targ_hist.append(dt[j,:,:])
                    cont_hist.append(dt[_idx!=j,:,:])
            end_idx = np.append(0,np.cumsum(cont_len).astype("int"))
            targ_hist = torch.stack(targ_hist)
            cont_hist = torch.cat(cont_hist,dim=0)
            pred[:,:,t] = self.forward(targ_hist, cont_hist, end_idx)[:,0:2]
            pred[:,:,t] = (Rot.transpose(1,2)@pred[:,:,t].view(-1,2,1)).view(-1,2) + Bias
            hist = torch.cat([hist[:,:,1:], pred[:,:,t].detach().view(-1,2,1)],dim=2)
        if fut_len==1: pred=pred.view(-1,2)
        return pred


    def predictNextLoc(self,hist,ei,n_guess,coef):
        n_traj, n_scene, hist_len, fut_len, = len(hist), len(ei)-1, self.ag.hist_len, self.ag.fut_len
        _d = hist.device
        Bias, Rot = torch.Tensor(n_traj,2).to(_d), torch.Tensor(n_traj,2,2).to(_d)

        targ_hist, cont_hist, cont_len, i_traj = [], [], [], -1
        for i in range(n_scene):
            data_scene = hist[ei[i]:ei[i+1],:,-hist_len:]
            n_ped = len(data_scene)
            cont_len = np.append(cont_len,[n_ped-1]*n_ped)
            _idx = np.arange(n_ped)
            for j in range(n_ped):
                i_traj+=1
                Bias[i_traj] = data_scene[j,:,-1]
                dt = data_scene - Bias[i_traj].view(-1,1)
                tht = pi/2+atan2(*dt[j,:,-2])
                Rot[i_traj] = torch.tensor([[cos(tht),-sin(tht)],[sin(tht),cos(tht)]])
                dt = (Rot[i_traj]@dt.transpose(0,1).reshape(2,-1)).view(2,-1,hist_len).transpose(0,1)
                targ_hist.append(dt[j,:,:])
                cont_hist.append(dt[_idx!=j,:,:])
        end_idx = np.append(0,np.cumsum(cont_len).astype("int"))
        targ_hist = torch.stack(targ_hist)
        cont_hist = torch.cat(cont_hist,dim=0)
        pred = self.forward(targ_hist, cont_hist, end_idx)
        netLoc_list = []
        for guess_i in range(n_guess):
            netLoc_list.append(utils.infLoc(pred,1,coef)[0])
        netLoc_list = torch.stack(netLoc_list)

        output = []
        for netLoc in netLoc_list:
            netLoc = (Rot.transpose(1,2)@netLoc.view(-1,2,1)).view(-1,2) + Bias
            output.append( torch.cat([hist, netLoc.detach().view(-1,2,1)],dim=2) )

        return torch.stack(output)


    def predictTrajSample(self,hist,ei=None):
        if ei is None: ei = [0,len(hist)]
        n_traj, n_scene, hist_len, fut_len, n_guess, n_sample = len(hist), len(ei)-1, self.ag.hist_len, self.ag.fut_len, self.ag.n_guess, self.ag.n_sample
        Hist = torch.stack([hist])
        coef = 1
        for t in range(fut_len):
            if len(Hist)>n_sample/n_guess:
                _idx = np.random.choice(len(Hist), int(n_sample/n_guess), replace=False)
                Hist = Hist[_idx]
            candidate_seq = []
            for i in range(len(Hist)):
                # print('Timestep ', t, ', sample', i, end='\r')
                _n_guess = n_sample if t==0 else n_guess
                coef *= self.ag.coef
                candidate_seq.append( self.predictNextLoc(Hist[i],ei,_n_guess,coef) )
            Hist = torch.stack(candidate_seq).view(-1,n_traj,2,hist_len+t+1)
        return Hist[:,:,:,-fut_len:]


    def lock_weights(self,lock=True):
        for layer_list in [self.t_conv, self.c_conv]:
            for layer in layer_list:
                layer.lock_weights(lock)



class L2Dist1d(nn.Module): # learning kerel seprately for x and y could save weights
    def __init__(self,n_ch, n_ker, ker_size):
        super(L2Dist1d, self).__init__()
        self.n_ch, self.n_ker, self.ker_size = n_ch, n_ker, ker_size
        self.weight = nn.Parameter(torch.Tensor(self.n_ker,self.n_ch,self.ker_size))
        self.bias = nn.Parameter(torch.zeros(self.n_ker))
        self.scaler = nn.Parameter(torch.ones(self.n_ker))
        self.weight_init() # self.weight.data.uniform_(-10, 10) #
        self.copier = nn.Parameter( torch.ones(1,self.n_ker).float() )
        self.copier.requires_grad = False

    def forward(self, x):
        shape = x.shape
        if len(shape)!=3 or shape[1]!=self.n_ch:
            print("Invalid input tensor",len(shape),shape[1],self.n_ch)
        batch_size, _, in_seq_len = shape
        out_seq_len = in_seq_len+1-self.ker_size
        if batch_size==0:
            return torch.zeros([0,self.weight.shape[0],out_seq_len])
        x = torch.nn.functional.unfold(x.view(-1,self.n_ch,in_seq_len,1),(self.ker_size,1)).transpose(1,2)
        x = x.view(batch_size,out_seq_len,-1,1).matmul(self.copier) - self.weight.view(self.n_ker,-1).t()
        # x = torch.log( x.pow(2).sum(2).sqrt().transpose(1,2) )
        x = torch.log( x.pow(2).view(batch_size,out_seq_len,self.n_ch,self.ker_size,self.n_ker).sum(2).sqrt().sum(2).transpose(1,2) )
        x = -x * torch.exp(self.scaler.view(-1,1)) + self.bias.view(-1,1)
        return x

    def weight_init(self):
        # if self.n_ch==2 and self.ker_size==1 and self.n_ker==7:
        #     self.weight.data = weight_init[:7] + torch.normal(0,.05,size=self.weight.shape)
        #     print("target conv[0] init.",self.weight.data.shape,self.weight.data.view(-1,2))
        # elif self.n_ch==2 and self.ker_size==1 and self.n_ker==17:
        #     self.weight.data = weight_init[:] + torch.normal(0,.05,size=self.weight.shape)
        #     print("context conv[0] init.",self.weight.data.shape,self.weight.data.view(-1,2))
        # if self.n_ch==2 and self.ker_size==2 and self.n_ker==13:
        if self.n_ch==2 and self.ker_size==2 and self.n_ker==len(weight_init):
            self.weight.data = weight_init[:]
            print("context conv[0] init.",self.weight.data.shape)
        else:
            self.weight.data.uniform_(-1, 1)

    def lock_weights(self,lock=True):
        if self.weight.requires_grad == lock:
            print("requires_grad",(not lock))
        self.weight.requires_grad = not lock
        # self.bias.requires_grad = not lock
        # self.scaler.requires_grad = not lock








    # def l2d_check(self, x):
    #     batch_size, _, in_seq_len = x.shape
    #     out_seq_len = in_seq_len+1-self.ker_size
    #     out = torch.zeros(batch_size,self.n_ker,out_seq_len).to(x.device)
    #     for i in range(batch_size):
    #         for j in range(self.n_ker):
    #             for k in range(out_seq_len):
    #                 sum = 0
    #                 for l in range(self.n_ch):
    #                     sum += (x[i,l,k:k+self.ker_size]-self.weight[j,l]).norm()**2
    #                 out[i,j,k] = torch.sqrt(sum)
    #     return out+self.bias.view(-1,1)


    # def conv(self,x):
    #     # out = torch.nn.functional.conv2d(x, self.weight)
    #     shape = x.shape
    #     if len(shape)!=3 or shape[1]!=self.n_ch:
    #         print("Invalid input tensor")
    #     batch_size, _, in_seq_len = shape
    #     out_seq_len = in_seq_len+1-self.ker_size
    #     x_unf = torch.nn.functional.unfold(x.view(-1, self.n_ch, in_seq_len,1),(self.ker_size,1)).transpose(1,2)
    #     out = x_unf.matmul(self.weight.view(self.n_ker,-1).t()).transpose(1,2).view(-1, self.n_ker, out_seq_len)
    #     out = out+self.bias.view(-1,1)
    #     return out
