import numpy as np
import xgboost as xgb

import torch
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 80,50

from sklearn import metrics
import cupy as cp

def log_regression_train(z_tra1,z_tra2, target_tra, z_val1, z_val2,target_val,args):
    torch.cuda.empty_cache()
    z_train = torch.cat((z_tra1, z_tra2), dim=1).detach().cpu().numpy()
    target_tra = target_tra.cpu().numpy()

    target_val = target_val.cpu().numpy()
    z_val = torch.cat((z_val1, z_val2), dim=1).detach().cpu().numpy()

    regressor = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.xg_lr,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree, 
        max_depth=args.max_depth,
        gamma=args.gamma,n_jobs=args.n_jobs,device=args.device,tree_method=args.tree_method)

    with cp.cuda.Device(args.gpu_ids[0]):
        regressor.fit(cp.array(z_train), cp.array(target_tra))
    return regressor

def log_regression_val(regressor, z_val1, z_val2, target_val, args):

    torch.cuda.empty_cache()
    target_val = target_val.cpu().numpy()
    z_val = torch.cat((z_val1, z_val2), dim=1).detach().cpu().numpy()

    with cp.cuda.Device(args.gpu_ids[0]):
        z_val = cp.array(z_val)
        y_pred = regressor.predict(z_val)
    mae = metrics.mean_absolute_error(target_val, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(target_val, y_pred))
    print("In log_regression_val:mae="+str(mae) + ',rmse='+str(rmse))

    # pickle.dump(regressor, open(str(mae)+"pima.pickle.dat", "wb"))
    
    return [mae,rmse], y_pred
