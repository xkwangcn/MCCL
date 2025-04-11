import copy
import torch
from torch import nn
from torchvision.models import resnet18,ResNet18_Weights

from einops import rearrange

from utils import *
    
class Classifier_One(nn.Module):

    def __init__(self, args):
        super(Classifier_One, self).__init__()

        self.args = args
        self.num_classes = args.num_classes
        self.device = torch.device('cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
        self.num_frames = args.num_frames
        self.instance_length = args.instance_length
        self.bag_size_video = self.num_frames // self.instance_length
        self.bag_size_cues = args.bag_size_cues

        # backbone
        model = resnet18(weights= ResNet18_Weights.DEFAULT) #
        print(model)
        self.features_2d = nn.Sequential(*list(model.children())[:-1])

        self.model = model
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, 512)
        self.features_2d_gray = self.model

        self.lstm = nn.LSTM(input_size=self.args.input_size_lstm, hidden_size=self.args.hidden_size_lstm, num_layers=self.args.num_layers_lstm, batch_first=True, bidirectional=True)

        # MHSA
        self.heads = self.args.head
        self.dim_head = self.args.hidden_size_lstm*2 // self.heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(self.args.hidden_size_lstm*2, (self.dim_head * self.heads) * 3, bias = False)

        self.norm = DMIN(num_features=self.args.hidden_size_lstm*2,args=self.args)
        self.pwconv_cues = nn.Conv1d(self.bag_size_cues, 1, 3, 1, 1)


    def lstm_mhsa(self, x):
        
        # [batch, bag_size, 512]
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        # [batch, bag_size, 256]
        ori_x = x # 32 4 128

        # MHSA
        qkv = self.to_qkv(x).chunk(3, dim=-1) # 32 4 128
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # q: 32,2,4,64;
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # 32 2 4 4 
        attn = self.attend(dots) # 32 2 4 4 
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)') # 32 2 4 64 - 32 4 128
        
        if self.bag_size_cues>1:
            x = self.norm(x)
        x = torch.sigmoid(x)
        
        x = ori_x * x

        return x, attn
    
    def forward(self, x_complete,epoch,mode):
        # 3d, gaze, pose, au
        features_list_cues = []
        features_types = [self.args.bag_cues_feature1,self.args.bag_cues_feature2,self.args.bag_cues_feature3,self.args.bag_cues_feature4,self.args.bag_cues_feature5]

        for i in range (0,self.bag_size_cues):
            if features_types[i]==3:
                x = x_complete[3] # B 1800 14
                x = x.unsqueeze(-1).permute(0, 3, 1, 2) # B 1800 14 1 # B 1 1800 14
                x = self.features_2d_gray(x).squeeze().unsqueeze(1) # 32 1 510
            else:
                index = features_types[i]
                x = x_complete[index].permute(0, 3, 1, 2) # 2017[B,T,F,3]-[B,3,T,F]
                x = self.features_2d(x).squeeze().unsqueeze(1) # B 512
            features_list_cues.append(x)

        cues_input_mo1 = torch.cat(features_list_cues, dim=1)
        
        features_tensor_cues, _ = self.lstm_mhsa(cues_input_mo1) # B bag_size self.args.hidden_size_lstm*2
        features_cues_conv= self.pwconv_cues(features_tensor_cues).squeeze() # [B, bag_size, 1024] ----  [batch, 1, 1024] .unsqueeze(1)
        torch.cuda.empty_cache()
        return features_cues_conv # 32 1 1024 | 32 256