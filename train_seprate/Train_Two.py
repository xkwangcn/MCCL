import torch
import copy
from torch import nn
from torchvision.models import resnet18,ResNet18_Weights
from einops import rearrange
from utils import *
    
class Classifier_Two(nn.Module):

    def __init__(self, args):
        super(Classifier_Two, self).__init__()

        self.args = args
        self.num_classes = args.num_classes
        self.device = torch.device('cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
        self.num_frames = args.num_frames
        self.instance_length = args.instance_length
        self.bag_size_video = self.num_frames // self.instance_length
        self.bag_size_cues = args.bag_size_cues

        # backbone
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        print(model)
        self.features_2d = nn.Sequential(*list(model.children())[:-1])

        self.model = model
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, 512)
        self.features_2d_gray = self.model

        self.lstm = nn.LSTM(input_size=self.args.input_size_lstm, hidden_size=self.args.hidden_size_lstm, num_layers=self.args.num_layers_lstm1, batch_first=True, bidirectional=True)

        # MHSA
        self.heads = self.args.head
        self.dim_head = self.args.hidden_size_lstm*2 // self.heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(self.args.hidden_size_lstm*2, (self.dim_head * self.heads) * 3, bias = False)

        self.norm = DMIN(num_features=self.args.hidden_size_lstm*2,args=self.args)
        self.pwconv_video = nn.Conv1d(self.bag_size_video, 1, 3, 1, 1)
        self.pwconv_second = nn.Conv1d(self.bag_size_cues, 1, 3, 1, 1)

    
    def lstm_mhsa(self, x):
        
        # [batch, bag_size, 512]
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        # [batch, bag_size, 1024]
        ori_x = x

        # MHSA
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots) # 32 2 3 3
        x = torch.matmul(attn, v) # 32 2 3 64, v=[32 2 3 64]
        x = rearrange(x, 'b h n d -> b n (h d)')

        if self.bag_size_video>1:
            x = self.norm(x)
        x = torch.sigmoid(x)
        
        x = ori_x * x

        return x, attn#(attn[:,0,:,:]+attn[:,1,:,:])/2
    
    def forward(self, x_complete,epoch,mode):
        features_list = []
        features_types = [self.args.bag_cues_feature1,self.args.bag_cues_feature2,self.args.bag_cues_feature3,self.args.bag_cues_feature4,self.args.bag_cues_feature5]
        
        for i in range (0,self.bag_size_cues):
            if features_types[i]==3:
                x = x_complete[3] # B 1440 14
                x = rearrange(x, 'b (t1 t2) f-> (b t1) t2 f', t1=self.bag_size_video, t2=self.instance_length) # 320 144 14
                # 320 144 14 1 - 320 1 144 14
                x = x.unsqueeze(-1).permute(0, 3, 1, 2) # B 1800 14 1 # B 1 1800 14
                x = self.features_2d_gray(x).squeeze(1) # 320 1 512
                x_be_mo1 = rearrange(x, '(b t1) f-> b t1 f',t1=self.bag_size_video) # 32 10 512
                af_mo1, _ = self.lstm_mhsa(x_be_mo1) # [B, bag_size, 1024]
                conv = self.pwconv_video(af_mo1).squeeze().unsqueeze(1) # [B, bag_size, 1024] ----  [batch, 1024]
            else:
                index = features_types[i]
                x = x_complete[index].permute(0, 1, 3, 2) #32 1260 68 3
                x = rearrange(x, 'b (t1 t2) c h-> (b t1) c t2 h', t1=self.bag_size_video, t2=self.instance_length) # 32 1260 3 68-> 96 3 420 68
                x = self.features_2d(x).squeeze() # [batch*bag_size, 512] 96 512
                x_be_mo1 = rearrange(x, '(b t) c -> b t c', t=self.bag_size_video)  # [batch, bag_size, 512] 32 3 512
                af_mo1, _ = self.lstm_mhsa(x_be_mo1) # [batch, bag_size, 1024] 32 3 128
                conv = self.pwconv_video(af_mo1).squeeze().unsqueeze(1) # [B, bag_size, 1024] ----  [batch, 1024]
            features_list.append(conv)
        features_tensor = torch.cat(features_list, dim=1)# B 4 1024
        features_tensor = self.pwconv_second(features_tensor).squeeze()# [B, bag_size, 1024] ----  [batch, 1024] .unsqueeze(1) 

        torch.cuda.empty_cache()
        return features_tensor # 32 256
