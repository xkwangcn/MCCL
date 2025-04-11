import argparse
import os
import datetime
import sys

from utils import Logger

class Options(object):
    """docstring for Options"""
    def __init__(self):
        super(Options, self).__init__()
        
    def initialize(self):

        parser = argparse.ArgumentParser()

        # settings
        parser.add_argument('--mode', type=str, default="train")
        parser.add_argument('--dataset', type=str, default="DAIC") # 
        parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids, eg. 0,1,2; -1 for cpu.')
        parser.add_argument('--device', type=str, default='cuda:1')
        parser.add_argument('--inference', type=str, default='0')
        parser.add_argument('--seed', default='42', type=int)
        parser.add_argument('--regressor_model', default='xgboost', type=str) # rf, xgboost,mlp
        parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
        parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
        parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N') #22
        parser.add_argument('--num_classes', default=1, type=int)

        # 2019 dataset settings
        parser.add_argument('--time_input', default=2048, type=int) # resnet and vgg dims
        parser.add_argument('--efeature_dl', default='resnet', type=str) # resnet, vgg
        parser.add_argument('--efeature_tr', default='openface', type=str) # bow, openface
        parser.add_argument('--efeature_audio', default='bow_mfcc', type=str) # choose video feature from : bow_mfcc, bow_egemaps, opensmile_mfcc, opensmile_egemaps
        parser.add_argument('--efeature_text', default='text', type=str) # enable 2019 data text modality
        parser.add_argument('--norm', default=0, type=int)

        # xgboost regressor setting
        parser.add_argument('--n_estimators', default='50', type=int)
        parser.add_argument('--xg_lr', default='0.08', type=float)
        parser.add_argument('--subsample', default='0.75', type=float)
        parser.add_argument('--colsample_bytree', default='1', type=float)
        parser.add_argument('--max_depth', default='4', type=int)
        parser.add_argument('--gamma', default='0', type=float)
        parser.add_argument('--n_jobs', default='-1', type=int)
        parser.add_argument('--tree_method', default='hist', type=str)
        
        # dims in encoders
        parser.add_argument('--input_size_lstm',default=512,type=int) # input dims
        parser.add_argument('--hidden_size_lstm',default=64,type=int) # final output dim = x*2
        parser.add_argument('--num_layers_lstm',default=1,type=int)
        parser.add_argument('--hidden_size_lstm1',default=32,type=int)
        parser.add_argument('--num_layers_lstm1',default=1,type=int)
        parser.add_argument('--num_layers_lstm2',default=1,type=int)
        # dims of cl loss
        parser.add_argument('--num_proj_hidden',default=128,type=int)

        parser.add_argument('--bag_size_cues',default=4,type=int) # feature number
        parser.add_argument('--bag_cues_feature1',default=0,type=int) # 0
        parser.add_argument('--bag_cues_feature2',default=1,type=int) # 1 gaze
        parser.add_argument('--bag_cues_feature3',default=2,type=int) # 2 pose
        parser.add_argument('--bag_cues_feature4',default=3,type=int) # 3 AUs
        parser.add_argument('--bag_cues_feature5',default=4,type=int) # optional

        # model settings
        parser.add_argument('--sample_num',default=10,type=int) #15
        parser.add_argument('--num_frames', default=12600//10, type=int, help='number of frames') # 12600//10,10800//10, 9390//10
        parser.add_argument('--instance_length', default=420, type=int, metavar='N', help='instance length') # 420 11 313
        parser.add_argument('--model', default='multi_cues_clip_fix', type=str, help='ideas') # multi_cues,multi_cues_clip, multi_level_icl,all_labels,multi_cues_clip_fix
        parser.add_argument('--main_mode', default='train_sep', type=str, help='') # train_sep, 
        parser.add_argument('--learn_mode', default='cl', type=str, help='') # 
        # training hyperparameters
        parser.add_argument('--head', default=2, type=int)
        parser.add_argument('--scale', default=-0.5, type=float)
        parser.add_argument('--tau', default=0.09, type=float)

        # optimizer
        parser.add_argument('-o', '--optimizer', default="AdamW", type=str, metavar='Opti')
        parser.add_argument('--lr', '--learning_rate', default=5e-4, type=float, metavar='LR', dest='lr') 
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
        parser.add_argument('--wd', '--weight_decay', default=0.05, type=float, metavar='W', dest='weight_decay')
        parser.add_argument('--eps', default=1e-1, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
        parser.add_argument('--label_smoothing', default=0.1, type=float, help='ratio of label smoothing')
        
        # scheduler
        parser.add_argument('--lr_scheduler', default="cosine", type=str)
        parser.add_argument('--warmup_epochs', default=10, type=int)
        parser.add_argument('--min_lr', default=5e-6, type=float)
        parser.add_argument('--warmup_lr', default=0, type=float)
        
        # setting in dataloader
        parser.add_argument('--dataset_path',default=r'/mnt/wd1/cv_dep/2017/wpingcheng/DAIC_WOZ-generated_database_V2/') #
        parser.add_argument('--output_path', default='./outputs', help='output path') # ./outputs
        parser.add_argument('--output_name', default='e1-h2-b8-s1-w0', help='output path') 
        parser.add_argument('--num_workers',default=0)
        
        parser.add_argument('--light', default=0, type=int)
        return parser

    def parse(self):

        parser = self.initialize()
        args = parser.parse_args()

        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            cur_id = int(str_id)
            if cur_id >= 0:
                args.gpu_ids.append(cur_id)

        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)

        # get logger
        file_name = os.path.join(
            args.output_path, '{}-{}.log'.format((datetime.datetime.now()),args.output_name))
        sys.stdout = Logger(file_name)
        for k in args.__dict__:
            print(k + ": " + str(args.__dict__[k]))

        return args