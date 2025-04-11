
import os
import pandas as pd
import torch
import numpy as np
import argparse

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset


class DepressionDataset(Dataset):

    def __init__(self, args,mode):
        '''
        input:
            dataset_path: /your_data_path/2017/wxk/latest
                feature_path /your_data_path/2017/wxk/latest/train/facial_keypoints2D
                label_path
                    /your_data_path/2017/wxk/latest/mode/_split_Depression_AVEC2017.csv
        return:
        '''
        self.args = args
        self.root_dir = os.path.join(args.dataset_path,mode,'original_data') # clipped_data
        self.sample_num = args.sample_num
    
        self.label = pd.read_csv(os.path.join(args.dataset_path, mode + '_split_Depression_AVEC2017.csv'))

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_index = [ j for j in range(0,1800,self.sample_num)] #180
    
        ID = int(self.label.loc[idx,'Participant_ID'])
        label = self.label.loc[idx,'PHQ8_Score']
        complete_label = self.label.loc[idx,:]
        binary = self.label.loc[idx,'PHQ8_Binary']
        fkps3d_path = os.path.join(self.root_dir, 'facial_keypoints')
        # 2. get gaze vectors feature
        gaze_path = os.path.join(self.root_dir, 'gaze_vectors')
        # 3. get pose
        pose_path = os.path.join(self.root_dir, 'position_rotation')
        AUs_path = os.path.join(self.root_dir, 'action_units')
        
        # clip_num = [0,2,4,6]
        clip_num = [0,1,2,3,4,5,6] #[0,1,2,3,4,5,6,7],[0,1]
        # 3*4
        one_cue_fkps_3d = []
        one_cue_gaze = []
        one_cue_pose = []
        one_cue_au = []
        for j in range(0,len(clip_num)):
            fkps_3d = torch.from_numpy(np.load(os.path.join(fkps3d_path,str(ID)+'-0'+str(clip_num[j])+'_kps.npy'))[sample_index,:]).type(torch.FloatTensor)
            gaze = torch.from_numpy(np.load(os.path.join(gaze_path, str(ID)+'-0'+str(clip_num[j])+'_gaze.npy'))[sample_index,:]).type(torch.FloatTensor)
            pose = torch.from_numpy(np.load(os.path.join(pose_path, str(ID)+'-0'+str(clip_num[j])+'_pose.npy'))[sample_index,:]).type(torch.FloatTensor)
            AUs = torch.from_numpy(np.load(os.path.join(AUs_path, str(ID)+'-0'+str(clip_num[j])+'_AUs.npy'))[sample_index,:]).type(torch.FloatTensor)

            one_cue_fkps_3d.append(fkps_3d)
            one_cue_gaze.append(gaze)
            one_cue_pose.append(pose)
            one_cue_au.append(AUs)
        
        one_cue_fkps_3d = torch.cat(one_cue_fkps_3d, dim=0)
        one_cue_gaze = torch.cat(one_cue_gaze, dim=0)
        one_cue_pose = torch.cat(one_cue_pose, dim=0)
        one_cue_au = torch.cat(one_cue_au, dim=0)
     
        return [one_cue_fkps_3d,
            one_cue_gaze,
            one_cue_pose,
            one_cue_au],torch.tensor(complete_label.values),binary
    
    def __len__(self):
        print('len: ',len(self.label))
        return len(self.label)

def get_dataloaders(args):

    dataloaders = {}
    for mode in ['train', 'validation']:
        dataset = DepressionDataset(args,mode)
        dataloaders[mode] = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,drop_last=False)
    return dataloaders


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',default=r'/your_data_path/2017/wpingcheng/DAIC_WOZ-generated_database_V2/')
    parser.add_argument('--batch_size',default=64)
    parser.add_argument('--num_workers',default=0)
    parser.add_argument('--sample_num',default=10)
    args = parser.parse_args()
    dataloaders = get_dataloaders(args)
    for i in dataloaders['train']: 
        print('loading')
