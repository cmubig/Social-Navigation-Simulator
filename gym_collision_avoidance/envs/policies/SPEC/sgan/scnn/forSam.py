import torch
import scnn.model as model
import scnn.utils as utils

#model = np.load("model.npy",allow_pickle=True)[0]

N = 23

hist = torch.arange(N*2*8).view(N,2,8)*1.0 #randomly generate some data in correct shape
fut = model.predictTraj(hist)
