import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os.path

_eps = 1e-12
_coef = 1.001
_tau = 0.001
gpu, cpu = "cuda:0", "cpu"
hist_len=8

tj = np.zeros([5,8,2])
tj[0] = [[0,0],[1,1],[2,2],[3,3],[4,4],[4,5],[4,6],[4,7]]
tj[1] = [[4,0],[4,1],[4,2],[4,3],[4,4],[4,5],[4,6],[4,7]]
tj[2] = [[6,0],[5.5,1],[5,2],[4.5,3],[4,4],[4,5],[4,6],[4,7]]
tj[3] = [[8,0],[7,1],[6,2],[5,3],[4,4],[4,5],[4,6],[4,7]]
tj[4] = [[8,0],[9,1],[10,2],[11,3],[12,4],[12,5],[12,6],[12,7]]
tj = tj.transpose(0,2,1)[:,:,:]
# print(tj.shape)

def getSample(n,m=tj.shape[0],l=8,eps=0.1):
    ind = np.random.randint(m,size=n)
    Smp = tj[:,:,:l][ind]
    Smp += np.random.uniform(-eps,eps,Smp.shape)
    return Smp,ind



def plotTraj_batch(batch):
    batch_size = len(batch[-1])
    for i in range(batch_size):
        start, end = batch[-1][i][0], batch[-1][i][1]
        hist_traj_list = batch[0][start:end]
        fut_traj_list = batch[1][start:end]
        plotTraj(hist_traj_list,fut_traj_list,fn=str(i))

def plotBatchTraj2(histTj, futTj, outTj, seq_start_end):
    for i in range(len(seq_start_end)):
        start, end = seq_start_end[i][0], seq_start_end[i][1]
        hist_traj_list = histTj[start:end]
        fut_traj_list = futTj[start:end]
        pred_list = outTj[start:end]
        print('\r',i,end=' ')
        plotTraj2(hist_traj_list,fut_traj_list,pred_list,fn=str(i))

def plotTraj2(hist_traj_list, fut_traj_list=None, pred_list=None, fn="000", path="./plots/"):
    n_ped, n_channel, hist_len = hist_traj_list.shape # n_channel is 2
    # plt.ion() # plt.show()
    plt.clf()
    if torch.is_tensor(hist_traj_list): hist_traj_list = hist_traj_list.numpy()
    if torch.is_tensor(fut_traj_list): fut_traj_list = fut_traj_list.numpy()
    for i in range(n_ped):
        plt.plot(hist_traj_list[i,0,:],hist_traj_list[i,1,:],'.-',color='C'+str(i%10))
        plt.plot(hist_traj_list[i,0,-1],hist_traj_list[i,1,-1],'*',color='C'+str(i%10))
        if fut_traj_list is not None:
            # plt.plot(fut_traj_list[i,0,:],fut_traj_list[i,1,:],':.',color='C'+str(i%10))
            plt.plot(np.append(hist_traj_list[i,0,-1],fut_traj_list[i,0,:]),np.append(hist_traj_list[i,1,-1],fut_traj_list[i,1,:]),':.',color='C'+str(i%10))
        if pred_list is not None:
            plt.plot(np.append(hist_traj_list[i,0,-1],pred_list[i,0,:]),np.append(hist_traj_list[i,1,-1],pred_list[i,1,:]),':*',color='C'+str(i%10))
    plt.axis("equal")
    plt.savefig(path+fn+".png")



def plotTraj(self_hist, self_fut, others_hist=np.array([]), others_fut=np.array([]), self_pred=np.array([]), fn="000", path="./plots/"):
    plt.clf()
    Traj = [others_hist, others_fut, self_hist, self_fut, self_pred]
    Mkr, Oth = ["-o", ":.", "-s", ":.", ":*"], [1,1,0,0,0]
    for i in range(5): # Convert to numpy.array
        if torch.is_tensor(Traj[i]): Traj[i] = Traj[i].detach().cpu().numpy()
        if Traj[i].size==0: Traj[i]=np.array([])
    len_hist, len_fut = int(Traj[2].size/2), int(Traj[3].size/2)
    for i in range(5):  # Reshape
        Traj[i] = Traj[i].reshape(-1,2,(len_hist if (i in [0,2]) else len_fut))
    for (i,j) in zip([1,3,4],[0,2,2]):
        if Traj[i].size:
            Traj[i] = np.append(Traj[j][:,:,-1].reshape(-1,2,1),Traj[i],axis=2)
    for i,(traj,mkr,oth) in enumerate(zip(Traj,Mkr,Oth)):
        for j in range(len(traj)):
            plt.plot(traj[j,0,:],traj[j,1,:], mkr, color='C'+str(j%9+oth))
    plt.plot([0],[0], ':.', color='k',label="Future Truth")
    plt.plot([0],[0], ':*', color='k',label="Future Prediction")
    plt.plot([0],[0], '-o', color='k',label="History")
    plt.legend()
    plt.axis("equal")
    plt.savefig(path+fn+".png")

def plotBatchTraj(th, tf, ch, cf, tp, ei):
    for i in range(len(th)):
        print('\r',i,end=' ')
        plotTraj(th[i],tf[i],ch[ei[i]:ei[i+1]],cf[ei[i]:ei[i+1]],tp[i],fn=str(i))
    print('.')


def getAde(fut,pred):
    if len(pred.shape)==4:
        err = (pred-fut).norm(dim=2).mean(dim=2).min(dim=0).values
        nan_err = torch.isnan(err)
        err[nan_err]=0
        return err.sum()/(len(err)-nan_err.sum())
        # return (pred-fut).norm(dim=2).mean(dim=2).min(dim=0).values.mean()
    if len(pred.shape)==2:
        if len(fut.shape)==2:
            fut, pred = fut.view(-1,2), pred.view(-1,2)
        else:
            fut, pred = fut[:,:,-1].view(-1,2), pred.view(-1,2)
    else:
        fut, pred = fut.transpose(1,2).reshape(-1,2), pred.transpose(1,2).reshape(-1,2)
    return (fut-pred).norm(dim=1).mean()

def getFde(fut,pred):
    if len(pred.shape)==4:
        err = (pred[:,:,:,-1]-fut[:,:,-1]).norm(dim=2).min(dim=0).values
        nan_err = torch.isnan(err)
        err[nan_err]=0
        return err.sum()/(len(err)-nan_err.sum())
        # return (pred[:,:,:,-1]-fut[:,:,-1]).norm(dim=2).min(dim=0).values.mean()
    if len(pred.shape)==2:
        fut, pred = fut[:,:,-1], pred[:,:2]
    else:
        fut, pred = fut[:,:,-1], pred[:,:,-1]
    return (fut-pred).norm(dim=1).mean()

def getBestSample(fut,pred,useAde=1):
    if useAde:
        idx = (pred-fut).norm(dim=2).mean(dim=2).argmin(dim=0)
    else:
        idx = (pred[:,:,:,-1]-fut[:,:,-1]).norm(dim=2).argmin(dim=0)
    return pred[ idx, torch.arange(len(idx)) ]


def getPdf(futLoc,pred):
    # https://mathworld.wolfram.com/BivariateNormalDistribution.html
    sx, sy, corr = torch.exp(pred[:,2]), torch.exp(pred[:,3]), torch.tanh(pred[:,4])

    normx = futLoc[:,0] - pred[:,0]
    normy = futLoc[:,1] - pred[:,1]
    sxsy = sx * sy
    z = (normx/sx+_eps)**2 + (normy/sy+_eps)**2 - 2*((corr*normx*normy)/sxsy+_eps)
    negRho = 1 - corr**2

    numer = torch.exp(-z/(2*negRho)) # Numerator
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))+_eps # Normalization factor
    pdf = numer/denom
    pdf = torch.clamp(pdf, min=_eps)

    return pdf


def getLoss(futLoc,pred):
    if pred.shape[1]==5:
        loss = -torch.log( getPdf(futLoc,pred) )
        return loss[~torch.isnan(loss)].mean()
    else:
        k0 = torch.sigmoid(pred[:,10])
        loss = -torch.log( k0*getPdf(futLoc,pred[:,:5]) + (1-k0)*getPdf(futLoc,pred[:,5:10]) )
        return loss[~torch.isnan(loss)].mean()

def getMeanCov(pred):
    mean = pred[:,0:2]
    sx, sy, corr = torch.exp(pred[:,2]), torch.exp(pred[:,3]), torch.tanh(pred[:,4])
    cov_xy = corr*sx*sy
    cov = torch.Tensor(len(pred),2,2).to(pred.device)
    cov[:,0,0], cov[:,1,1] = (sx**2+_tau)*_coef, (sy**2+_tau)*_coef
    cov[:,0,1], cov[:,1,0] = cov_xy, cov_xy
    return mean, cov

def infLoc(pred,n=1,coef=1.0):
    if pred.shape[1]!=5:
        batch_size = pred.shape[0]
        k0 = torch.sigmoid(pred[:,10]).to(cpu)
        _idx = [range(batch_size),(torch.rand([batch_size]) < k0).int().tolist()]
        pred = pred[:,:10].view(-1,2,5)[_idx]
    # mean, cov = getMeanCov(pred) # to replace the next 6 lines
    mean = pred[:,0:2]
    sx, sy, corr = torch.exp(pred[:,2]), torch.exp(pred[:,3]), torch.tanh(pred[:,4])
    cov_xy = corr*sx*sy
    cov = torch.Tensor(len(pred),2,2).to(pred.device)
    cov[:,0,0], cov[:,1,1] = (sx**2+_tau)*_coef, (sy**2+_tau)*_coef
    cov[:,0,1], cov[:,1,0] = cov_xy, cov_xy
    cov *= coef
    try:
        sample = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov).sample([n])
    except:
        print("failed")
        N = len(mean)
        sample = torch.Tensor(n,N,2)
        for i in range(N):
            try:
                sample[:,i,:] = torch.distributions.multivariate_normal.MultivariateNormal(mean[i], cov[i]).sample([n])
            except:
                print(i,pred[i],mean[i],cov[i])
                while True: pass
    # if torch.isnan(sample).sum()!=0:
    #     for k in range(len(sample)):
    #         i = torch.where(torch.isnan(sample[k]))[0]
    #         i = list(set(i.tolist()))
    #         print('\n\n\n',i,'\n',sample[k][i],'\n',pred[i],'\n',cov[i],'\n',sx[i],'\n',sy[i],'\n',corr[i],'\n',cov[i])
    #         sample[k,i] = pred[i,:2]
    #         print('\n',sample[k][i])
    return sample


def printParamGrad(net):
    params = list(net.parameters())
    for i in range(len(params)):
        p = params[i].grad.abs()
        print( p.mean().item(),p.max().item() )
        # print( '\t%.5f\t%.5f' % (p.mean().item(),p.max().item()) )
    #

def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    _dir = '/home/dzhao/proj/scnn'
    dir_path = os.path.join(_dir, 'datasets', dset_name, dset_type)
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)
    return dir_path


def plot_trainBatch(b,tp=None):
    th, tf = b[0], b[3].reshape(-1,2,1)
    emp = np.empty([len(th),0])
    if tp is None: tp = emp
    ch, cf = b[1], emp
    end_idx = b[2]
    plotBatchTraj(th,tf,ch,cf,tp,end_idx)

def plot_valBatch(ei,hist,fut=None,pred=None,fn='val_',num=None,amount=0.3):
    for i in range(len(ei)-1):
        h = hist[ei[i]:ei[i+1]]
        f = None if fut is None else fut[ei[i]:ei[i+1]]
        print('\r',i,end=' ')
        if (pred is not None and len(pred.shape)==4):
            p = None if pred is None else pred[:,ei[i]:ei[i+1]]
            plotTraj3(h, f, p, fn=fn+str(i), path="./plots/",amount=amount)
        else:
            p = None if pred is None else pred[ei[i]:ei[i+1]]
            plotTraj2(h, f, p, fn=fn+str(i), path="./plots/")
        if num is not None and i>=num: break
    print('.'+fn)

def lighten_color(color, amount=0.3):
    import matplotlib.colors as mc
    import colorsys
    c = colorsys.rgb_to_hls(*mc.to_rgb(color))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def plotTraj3(hist_traj_list, fut_traj_list=None, pred_list_20=None, fn="000", path="./plots/", amount=0.3):
    n_ped, n_channel, hist_len = hist_traj_list.shape # n_channel is 2
    # plt.ion() # plt.show()
    plt.clf()
    if torch.is_tensor(hist_traj_list): hist_traj_list = hist_traj_list.numpy()
    if torch.is_tensor(fut_traj_list): fut_traj_list = fut_traj_list.numpy()
    for i in range(n_ped):
        _c = lighten_color('C'+str(i%10),amount)
        for j in range(len(pred_list_20)):
            pred_list = pred_list_20[j]
            plt.plot(np.append(hist_traj_list[i,0,-1],pred_list[i,0,:]),np.append(hist_traj_list[i,1,-1],pred_list[i,1,:]),':*',color=_c)
    for i in range(n_ped):
        plt.plot(hist_traj_list[i,0,:],hist_traj_list[i,1,:],'.-',color='C'+str(i%10))
        plt.plot(hist_traj_list[i,0,-1],hist_traj_list[i,1,-1],'*',color='C'+str(i%10))
    for i in range(n_ped):
        if fut_traj_list is not None:
            # plt.plot(fut_traj_list[i,0,:],fut_traj_list[i,1,:],':.',color='C'+str(i%10))
            plt.plot(np.append(hist_traj_list[i,0,-1],fut_traj_list[i,0,:]),np.append(hist_traj_list[i,1,-1],fut_traj_list[i,1,:]),':.',color='C'+str(i%10))
    for i in range(n_ped):
        best_sample_id = np.linalg.norm(pred_list_20[:,i]-fut_traj_list[i],axis=1).mean(axis=1).argmin()
        for j in range(len(pred_list_20)):
            if j!=best_sample_id: continue
            pred_list = pred_list_20[j]
            plt.plot(np.append(hist_traj_list[i,0,-1],pred_list[i,0,:]),np.append(hist_traj_list[i,1,-1],pred_list[i,1,:]),':*',color='C'+str(i%10))
    plt.axis("equal")
    plt.savefig(path+fn+".png")


def save_model(predictor,fn):
    np.save("model/"+fn,[predictor])

def load_model(fn):
    if fn[-4:]!=".npy": fn+=".npy"
    return np.load("model/"+fn,allow_pickle=True)[0]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def update_lr(epoch, lr):
    # thres, LR = [0.0, 0.0, 0.42, 0.46, 0.50, 0.55, 0.6, 0.7], 0.00001*np.array([0.01,0.03,0.1,0.3,1,3,10,30])
    # for i in range(len(thres)):
    #     if loss<thres[i] and LR[i]<=lr: return LR[i]
    Epoch, LR = [250, 200, 150, 100, 50, 20, 10], 0.00001*np.array([0.03,0.1,0.3,1,3,10,30])
    for i in range(len(Epoch)):
        if epoch>Epoch[i] and LR[i]<=lr: return LR[i]
    return lr


def eval_model(predictor, loader_trj, determ=1, n_batch=1, fn='', useAde=0, repeat=False):
    if repeat:
        N,A,D=0,0,0
        if 1: #try:
            for i in range(16):
                a,d = eval_model(predictor, loader_trj, determ=determ, n_batch=n_batch, fn=fn, useAde=useAde)
                A+=a
                D+=d
                N+=1
                print('%.3f/%.3f %.3f/%.3f looping'%(a,d,A/N,D/N))
        # except:
        #     print('Ave: %.3f/%.3f'%(A/N,D/N))
        return A/N, D/N
    ade, fde = 0, 0
    for i, b in enumerate(loader_trj):
        if i>=n_batch: break
        hist, fut, ei = b[0][:,:,:hist_len], b[0][:,:,hist_len:], b[1]
        if determ:
            pred = predictor.predictTraj(hist.to(gpu),ei).to(cpu)
        else:
            pred = predictor.predictTrajSample(hist.to(gpu),ei).to(cpu)
        ade += getAde(fut, pred)
        fde += getFde(fut, pred)
        # print('\r',i,end='')
    n_batch = min(i+1,n_batch)
    ade, fde = ade/n_batch, fde/n_batch
    if fn:
        if not determ: pred = getBestSample(fut,pred,useAde=useAde)
        plot_valBatch( ei, hist.numpy(), fut.numpy(), pred.detach().numpy(),fn=fn, num=100 )
    return ade.item(), fde.item()


def write_csv(line,fn=''):
    import csv
    with open('result/'+fn+'_error.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(line)

def plot_err(err,ade_t,fde_t,fn):
    plt.figure()
    clr = ['C1','C5','C6','C4','C0','C9']
    leg = ['a_td','f_td','a_vd','f_vd','a_vp','f_vp']
    err = np.array(err)
    N, M = err.shape
    for i in range(M):
        plt.plot(err[:,i], '.-', color=clr[i],label=leg[i])
    plt.plot([0,N],[ade_t,ade_t],'k')
    plt.plot([0,N],[fde_t,fde_t],'k')
    plt.legend()
    plt.savefig("./plots/err_"+fn+".png")

def remap(t,scale=1):
    t -= t.min()
    t /= t.max()
    t = t*2*scale-scale
    t = torch.tanh(t)
    t -= t.min()
    t /= t.max()
    return t*2-1

def heat2rgb(t):
    def clamp(v): return max(0,min(v,1))
    r = clamp(1.5 - abs(2.0*t-1.0))
    g = clamp(1.5 - abs(2.0*t))
    b = clamp(1.5 - abs(2.0*t+1.0))
    return (r,g,b)

def plotWeights(P, W=None, targ_traj=None, Traj=None, fn="000", path="./plots/", lim=None, _c=None,_a=None):
    plt.clf()
    fig = plt.figure(figsize=[8,8])
    plt.plot(0,0,"^")
    if torch.is_tensor(P): P = P.numpy().copy()
    plt.plot([np.min(P[:,1,:]),np.max(P[:,1,:])],[np.min(P[:,0,:]),np.max(P[:,0,:])],'w.')
    if W is None: W = -np.ones(len(P)) # np.arange(len(P))/len(P)*2-1# W = np.random.random(len(P))
    for (p,w) in zip(P,W):
        p[:,1]-=p[:,0]
        plt.arrow(p[1,0], p[0,0], p[1,1], p[0,1], length_includes_head=True, color=heat2rgb(w) if _c is None else _c, alpha=1.0 if _a is None else _a, width=.03)
    if targ_traj is not None:
        if torch.is_tensor(targ_traj): targ_traj = targ_traj.numpy()
        plt.plot(targ_traj[1,:],targ_traj[0,:],'^-',color='C0')
    if Traj is not None:
        if torch.is_tensor(Traj): Traj = Traj.numpy()
        Traj = Traj.reshape(-1,2,8)
        for i,traj in enumerate(Traj):
            plt.plot(traj[1,:],traj[0,:],'.-',color='C'+str((i+1)%10))
            plt.plot(traj[1,-1],traj[0,-1],'o',color='C'+str((i+1)%10))
    if lim is None: plt.axis("equal")
    else: plt.xlim(lim[0],lim[1]); plt.ylim([lim[2],lim[3]])
    plt.savefig(path+fn+".png")
