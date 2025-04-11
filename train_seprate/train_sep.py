import pickle
import cupy as cp
import torch
import torch.nn as nn
import os

import numpy as np
from ContrastiveLoss import CustomSCLLoss
from train_seprate.Train_One import Classifier_One
from train_seprate.Train_Two import Classifier_Two
from train_seprate.regression import log_regression_train, log_regression_val

from dataloaders.depression_dataset import get_dataloaders
from sklearn import metrics
from utils import *
torch.autograd.set_detect_anomaly(True)

def create_model(args):
    model1 = Classifier_One(args)
    model2 = Classifier_Two(args)
    return model1, model2

class Solver_Sep(object):
    def __init__(self, args):
        super(Solver_Sep, self).__init__()

        self.args = args

        # init cuda
        if len(self.args.gpu_ids) > 0:
            torch.cuda.set_device(self.args.gpu_ids[0])
        self.device = torch.device('cuda:%d' % self.args.gpu_ids[0] if self.args.gpu_ids else 'cpu')
        
        # set seed
        seed = self.args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        # init dataloader
        if args.dataset == 'DAIC':
            dataloaders = get_dataloaders(args)
            self.train_dataloader = dataloaders['train']
            self.test_dataloader = dataloaders['validation']
        self.mse = nn.MSELoss().to(self.device)

        self.model1,self.model2 = create_model(self.args)
        
        self.model1.to(self.device)
        self.model2.to(self.device)

        self.two_scl_loss = CustomSCLLoss(args).to(self.device)

        # init optimizer and scheduler
        self.optimizer1 = torch.optim.AdamW(self.model1.parameters(), lr=self.args.lr, eps=self.args.eps, weight_decay=self.args.weight_decay)
        self.scheduler1 = build_scheduler(self.args, self.optimizer1, len(self.train_dataloader))

        self.optimizer2 = torch.optim.AdamW(self.model2.parameters(), lr=self.args.lr, eps=self.args.eps, weight_decay=self.args.weight_decay)
        self.scheduler2 = build_scheduler(self.args, self.optimizer2, len(self.train_dataloader))


    def run(self):

        best_global_val_mae = 100
        best_result = [100,100] # mae, rmse
        if self.args.inference == '1':
            self.model1 = torch.load(os.path.join('checkpoint/',self.args.dataset,'current_model1'))
            self.model2 = torch.load(os.path.join('checkpoint/',self.args.dataset,'current_model2'))
            val_result,regressor,val_loss = validate(self.model1, self.model2, self.train_dataloader, self.test_dataloader, self.args,276)
            return
        for epoch in range(self.args.start_epoch+1, self.args.epochs+1):
            inf = '********************' + str(epoch) + '********************'
            print(inf)
            train_loss = self.train(epoch)
            val_result,regressor,val_loss = validate(self.model1, self.model2, self.train_dataloader, self.test_dataloader, self.args,epoch)
            print (f'val_mae={val_result[0]:.4f}')

            if best_global_val_mae > val_result[0]:
                best_global_val_mae = val_result[0]
                best_result = val_result
                save_path = os.path.join(self.args.output_path,self.args.dataset)
                if not os.path.exists(save_path): os.mkdir(save_path)
                torch.save(self.model1,os.path.join(save_path,'current_model1'))
                torch.save(self.model2,os.path.join(save_path,'current_model2'))
                print("will update:mae="+str(best_result[0]) + ',rmse='+str(best_result[1])+'\n')
                                
            print("Currebt best mae="+str(best_result[0])+",rmse="+str(best_result[1]))
                  
        print("Final best val mae="+str(best_result[0])+",rmse="+str(best_result[1]))
                
    def train(self, epoch):
        all_loss = 0
        self.model1.train()
        self.model2.train()
        for i, (features, target,binary) in enumerate(self.train_dataloader):
            torch.cuda.empty_cache()
            print("Training epoch \t{}: {}\\{}".format(epoch, i + 1, len(self.train_dataloader)), end='\r')
            for v in range(len(features)):
                features[v] = features[v].to(self.device)
            target = target.to(self.device).to(torch.float32)
            if self.args.dataset == 'DAIC':
                target = target[:,2]

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()

            #####################train with two stage##############################
            feature_tensor1 = self.model1(features,epoch,'train') # 32 256
            feature_tensor2 = self.model2(features,epoch,'train') # 32 256
            loss = self.two_scl_loss(feature_tensor1,feature_tensor2).to(torch.float32)
            all_loss = all_loss + loss

            loss.backward()
            self.optimizer1.step()
            self.scheduler1.step_update(epoch * len(self.train_dataloader) + i)
            self.optimizer2.step()
            self.scheduler2.step_update(epoch * len(self.train_dataloader) + i)

        loss = all_loss / len(self.train_dataloader)
        print('train loss: ' + str(loss))

        return loss

    
def validate (model1, model2, tra_dataloader, val_dataloader, args,epoch):
    two_scl_loss = CustomSCLLoss(args).to(args.device)
    with torch.no_grad():
        model1.eval()
        model2.eval()
        rep_tra1_all = []
        rep_tra2_all = []
        rep_val1_all = []
        rep_val2_all = []
        target_tra_all = []
        target_val_all = []

        if args.inference=='0':
            for i, (fea_tra_ori, target_tra, _) in enumerate(tra_dataloader):
                torch.cuda.empty_cache()
                for v in range(len(fea_tra_ori)):
                    fea_tra_ori[v] = fea_tra_ori[v].to(args.device)
                target_tra = target_tra.to(args.device).to(torch.float32)
                if args.dataset == 'DAIC':
                    target_tra = target_tra[:,2]

                rep_tra1 = model1(fea_tra_ori,epoch,'val_trainset') # 32 1024
                rep_tra2 = model2(fea_tra_ori,epoch,'val_trainset') # 32 1024

                rep_tra1_all.append(rep_tra1)
                rep_tra2_all.append(rep_tra2)
                target_tra_all.append(target_tra)

            rep_tra1_all = torch.cat(rep_tra1_all, dim=0)
            rep_tra2_all = torch.cat(rep_tra2_all, dim=0)
            target_tra_all = torch.cat(target_tra_all,dim = 0)

        all_loss =0
        for i, (fea_val_ori, target_val, _) in enumerate(val_dataloader):
            torch.cuda.empty_cache()
            for v in range(len(fea_val_ori)):
                fea_val_ori[v] = fea_val_ori[v].to(args.device)
            target_val = target_val.to(args.device).to(torch.float32)
            if args.dataset == 'DAIC':
                target_val = target_val[:,2]

            rep_val1 = model1(fea_val_ori,epoch,'val') # 32 1024
            rep_val2 = model2(fea_val_ori,epoch,'val') # 32 1024

            loss = two_scl_loss(rep_val1,rep_val2).to(torch.float32)
            all_loss = all_loss + loss
            rep_val1_all.append(rep_val1)
            rep_val2_all.append(rep_val2)
            target_val_all.append(target_val)
            
        rep_val1_all = torch.cat(rep_val1_all, dim=0)
        rep_val2_all = torch.cat(rep_val2_all, dim=0)
        target_val_all = torch.cat(target_val_all,dim = 0)
        loss = all_loss / len(val_dataloader)
        
        if args.inference=='1':
            with open(os.path.join('checkpoint/',args.dataset,"pima.pickle.dat"),'rb') as f:  
                torch.cuda.empty_cache()
                target_val_all = target_val_all.cpu().numpy()
                z_val = torch.cat((rep_val1_all, rep_val2_all), dim=1).detach().cpu().numpy()
                with cp.cuda.Device(args.gpu_ids[0]):
                    z_val = cp.array(z_val)
                    clf_load = pickle.load(f)
                    y_pred = clf_load.predict(z_val)
                    
                mae = metrics.mean_absolute_error(target_val_all, y_pred)
                rmse = np.sqrt(metrics.mean_squared_error(target_val_all, y_pred))
                print("In log_regression_val:mae="+str(mae) + ',rmse='+str(rmse))

                return [mae,rmse], clf_load,loss

    regressor = log_regression_train(rep_tra1_all, rep_tra2_all, target_tra_all,rep_val1_all, rep_val2_all, target_val_all, args)
    val_result,pred = log_regression_val(regressor, rep_val1_all, rep_val2_all, target_val_all,args)

    return val_result,regressor,loss
