import torch
import torch.nn as nn
import torch.nn.functional as F

class CVM:
    def rel_to_abs(self,rel_traj, start_pos):
        rel_traj = rel_traj.permute(1, 0, 2)
        displacement = torch.cumsum(rel_traj, dim=1)
        start_pos = torch.unsqueeze(start_pos, dim=1)
        abs_traj = displacement + start_pos
        return abs_traj.permute(1, 0, 2)

    def constant_velocity_model(self,observed, sample=False):
        """
        CVM can be run with or without sampling. A call to this function always
        generates one sample if sample option is true.
        """
        obs_rel = observed[1:] - observed[:-1]
        deltas = obs_rel[-1].unsqueeze(0)
        if sample:
            sampled_angle = np.random.normal(0, 25, 1)[0]
            theta = (sampled_angle * np.pi)/ 180.
            c, s = np.cos(theta), np.sin(theta)
            rotation_mat = torch.tensor([[c, s],[-s, c]]).to(gpu)
            deltas = torch.t(rotation_mat.matmul(torch.t(deltas.squeeze(dim=0)))).unsqueeze(0)
        y_pred_rel = deltas.repeat(12, 1, 1)
        return y_pred_rel

    def predictTraj(self,hist,ei=None):
        if ei is None: ei = [0,len(hist)]
        observed = hist.permute(2,0,1)
        y_pred_rel = self.constant_velocity_model(observed)
        y_pred_abs = self.rel_to_abs(y_pred_rel, observed[-1])
        return y_pred_abs.permute(1,2,0)

    def predictTrajSample(self,hist,ei=None):
        if ei is None: ei = [0,len(hist)]
        print('sample',end=',')
        observed = hist.permute(2,0,1)
        y_samples = []
        for i in range(20):
            print(i,end='\r')
            y_pred_rel = self.constant_velocity_model(observed,sample=True)
            y_pred_abs = self.rel_to_abs(y_pred_rel, observed[-1])
            y_samples.append(y_pred_abs.permute(1,2,0))
        return torch.stack(y_samples)




model = CVM()
N = 23
hist = torch.arange(N*2*8).view(N,2,8)*1.0 #randomly generate some data in correct shape
fut = model.predictTraj(hist)

print("hist")
print(hist)
print("fut")
print(fut)
