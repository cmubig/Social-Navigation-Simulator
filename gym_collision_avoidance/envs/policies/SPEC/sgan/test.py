import torch
import scnn.model as model
# import scnn.utils as utils
import numpy as np
model = np.load("univ_best_1.npy",allow_pickle=True)[0]
N = 23
hist = torch.arange(N*2*8).view(N,2,8)*1.0 #randomly generate some data in correct shape
import time



start_time = time.time()
fut = model.predictTraj(hist.to("cuda"))
print("time = "+str(time.time()-start_time)) # time = 0.13746118545532227
print(fut.shape)



start_time = time.time()
fut = model.predictTrajSample(hist.to("cuda"))[0]
print("time = "+str(time.time()-start_time)) # time = 2.864546775817871
print(fut.shape)



start_time = time.time()
fut = model.predictNextLoc(hist.to("cuda"))[0,:,:,-1] # (23,2,12)
print("time = "+str(time.time()-start_time)) # time = 1.5792806148529053
print(fut.shape)

start_time = time.time()
fut = model.predictNextLoc(hist.to("cuda"))[0,:,:,-1] # (23,2,12)
print("time = "+str(time.time()-start_time)) # time = 1.5792806148529053
print(fut.shape)
