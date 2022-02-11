import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

print(torch.cuda.is_available())
device=torch.device("cpu")

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class PatNet(nn.Module):
    ## HYPERPARAMTERS
    def __init__(self, nA, nS):
        super(PatNet,self).__init__()
        self.nA=nA
        self.input_dims=nS
        self.fc1=nn.Linear(self.input_dims, 128)    # 1st Hidden Layer
        self.fc2=nn.Linear(128, 84)                # 2nd Hidden Layer
        self.fc3=nn.Linear(84, self.nA)            # Output Layer
        # self.dropout=nn.Dropout(p=0.2)              # Dropout

    def forward(self, observation):
        observation=F.relu(self.fc1(observation))
        observation=F.relu(self.fc2(observation))
        qsa=self.fc3(observation)

        return qsa

class DefNet(nn.Module):
    def __init__(self):
        pass

Q_policy=PatNet(4, 6)
# Q_policy=DefNet()

print("tested")