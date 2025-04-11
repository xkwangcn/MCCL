from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomSCLLoss(nn.Module):
    def __init__(self, args):
        super(CustomSCLLoss, self).__init__()

        self.args = args
        num_hidden = args.hidden_size_lstm*2
        num_proj_hidden = args.num_proj_hidden

        self.tau = args.tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) - refl_sim.diag() + between_sim.sum(1)))  # sum(1) 行求和

    def forward(self, f1, f2):
        f1 = self.projection(f1)
        f2 = self.projection(f2)

        l1 = self.semi_loss(f1, f2)
        l2 = self.semi_loss(f2, f1)
        loss = (l1 + l2) * 0.5
        loss = loss.mean()

        return loss
    