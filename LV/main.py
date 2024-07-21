"""
Author: Minglang Yin, myin16@jhu.edu

Branch Nets:             [pace_uvc; geo_mode]
Trunk Net:              [uvc]
Output:                 [ATs/RTs]
"""

import os
import torch
import time
import numpy as np
import scipy.io as io
import math

from utils import *
from opnn import *

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

def main():
    ## hyperparameters
    args = ParseArgument()
    epochs = args.epochs
    device = args.device
    save_step = args.save_step
    test_model = args.test_model
    if test_model == 1:
        load_path = "./CheckPts/model_chkpts.pt"

    num_geomode = 180
    if_normal = 1

    dim_br_geo = [num_geomode, 200, 200, 200, 200]
    dim_br_pace = [2, 200, 200, 200, 200]
    dim_tr = [3, 200, 200, 200, 200] ## need to try 4 later

    num_train = 900
    num_test = 106
    num_cases = num_train + num_test
    
    ## create folders
    dump_train = './Predictions/Train/'
    dump_test = './Predictions/Test/'
    model_path = 'CheckPts/model_chkpts.pt'
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)
    
    ## dataset
    datafile = "../data_train_h5/LATs_1006.npy"
    with open(datafile, 'rb') as handle:
        dataset = pickle.load(handle)

    u_train = dataset['RTs_train']
    u_train = np.transpose(u_train, (0, 2, 1))
    surf_train = dataset["coeff_train"] 
    x_pace_train = dataset["x_pace_train"]

    u_test = dataset['RTs_test']
    u_test = np.transpose(u_test, (0, 2, 1)) # shape: [num_case, num_pace, num_uvc]
    surf_test = dataset["coeff_test"]
    x_pace_test = dataset["x_pace_test"]

    x_data = dataset['x']
    x_scale = dataset["x_scale"]
    u_scale = 100

    ################
    ## set pace on epi #######
    x_pace_train = np.concatenate((x_pace_train[:, :, 0:1], x_pace_train[:, :, 2:3]), axis=2)
    x_pace_test = np.concatenate((x_pace_test[:, :, 0:1], x_pace_test[:, :, 2:3]), axis=2)

    ### downsampling
    nskip = 10
    x_data = x_data[::nskip]
    u_test = u_test[:, :, ::nskip]
    u_train = u_train[:, :, ::nskip]

    ##############
    num_pts = x_data.shape[0]
    u_test_reshape_phy = u_test.reshape(-1, num_pts)
    u_train_reshape_phy = u_train.reshape(-1, num_pts)

    ## normalization
    if if_normal == 1:
        surf_mean = surf_train.mean(axis=0)
        surf_std = surf_train.std(axis=0)
        surf_train = (surf_train - surf_mean)/surf_std
        surf_test = (surf_test - surf_mean)/surf_std

        u_test_phy = u_test.copy()
        u_mean_train = u_train_reshape_phy.min()
        u_std = 1

        u_train = (u_train - u_mean_train)/u_std
        u_test = (u_test - u_mean_train)/u_std

    ## tensor
    u_train_tensor = torch.tensor(u_train, dtype=torch.float).to(device)
    f_train_tensor = torch.tensor(surf_train, dtype=torch.float).to(device)
    x_pace_train_tensor = torch.tensor(x_pace_train, dtype=torch.float).to(device)

    f_test_tensor = torch.tensor(surf_test, dtype=torch.float).to(device)
    x_pace_test_tensor = torch.tensor(x_pace_test, dtype=torch.float).to(device)
    
    x_tensor = torch.tensor(x_data, dtype=torch.float).to(device)

    ## initialization
    model = opnn(dim_br_geo, dim_br_pace, dim_tr).to(device)
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)

    if test_model == 0:
        train_loss = np.zeros((args.epochs, 1))
        test_loss = np.zeros((args.epochs, 1))
        mean_abs_err = np.zeros((args.epochs, 1))
        rel_MSE_err = np.zeros((args.epochs, 1))

        ## training
        def train(epoch, f, x_pace, x, y):
            model.train()
            def closure():
                optimizer.zero_grad()
                y_pred = model.forward(f, x_pace, x)
                loss = ((y_pred - y)**2).mean()
                train_loss[epoch, 0] = loss
                loss.backward()
                return loss
            optimizer.step(closure)

        ## Iterations
        print('start training...', flush=True)
        tic = time.time()
        for epoch in range(0, epochs):
            if epoch == 20000:
                optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
            # elif epoch == 90000:
            #     optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
            # elif epoch == epochs - 1000:
            #     optimizer = torch.optim.LBFGS(model.parameters())

            ## Training
            train(epoch, f_train_tensor, x_pace_train_tensor, x_tensor, u_train_tensor)

            ## Testing
            y_pred = to_numpy(model.forward(f_test_tensor, x_pace_test_tensor, x_tensor))
            loss_tmp = ((y_pred - u_test)**2).mean() # denote u_test as testing cardiac field
            test_loss[epoch, 0] = loss_tmp

            ## errors
            y_pred_phy = y_pred*u_std + u_mean_train
            mean_abs_err[epoch] = (abs(y_pred_phy - u_test_phy)*u_scale).mean()
            
            rel_MSE_err[epoch] = np.mean(
                                        np.linalg.norm(u_test_reshape_phy - y_pred_phy.reshape(-1, num_pts), axis=1)/np.linalg.norm(u_test_reshape_phy, axis=1)
                                        )

            ## testing error
            if epoch%100 == 0:
                print(f'Epoch: {epoch}, Train Loss: {train_loss[epoch, 0]:.6f}, Test Loss: {test_loss[epoch, 0]:.6f}, Rel L2 Err: {rel_MSE_err[epoch, 0]:.6f}, Mean Abs Loss: {mean_abs_err[epoch, 0]:.6f}', flush=True)
                
            ## Save model
            if (epoch+1)%save_step == 0: 
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, model_path)

        toc = time.time()

        print(f'total training time: {int((toc-tic)/60)} min', flush=True)
        np.savetxt('./train_loss.txt', train_loss)
        np.savetxt('./test_loss.txt', test_loss)
        np.savetxt('./mean_abs_err.txt', mean_abs_err)
        np.savetxt('./rel_MSE_err.txt', rel_MSE_err)

        ## plot loss function
        num_epoch = train_loss.shape[0]
        x = np.linspace(1, num_epoch, num_epoch)
        
        ## plot loss
        fig = plt.figure(constrained_layout=False, figsize=(6, 6))
        gs = fig.add_gridspec(1, 1)

        ax = fig.add_subplot(gs[0])
        ax.plot(x, train_loss.mean(axis=1), color='blue', label='Training Loss')
        ax.plot(x, test_loss.mean(axis=1), color='red', label='Test Loss', linestyle='dashed')
        ax.set_yscale('log')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epochs')
        ax.legend(loc='upper left')
        fig.savefig('./loss_his.png')

    else:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        ## save for visualization
        vis_case_list = ["HCM_new_696", "HCM_new_706"]
        vis_test_id = [38, 48]
        for id, vis_case in enumerate(vis_case_list):
            x_vis = np.loadtxt("./vis/" + vis_case + "/uvc.dat")
            num_length = x_vis.shape[0]
            save_id = vis_test_id[id]
            
            ## predict in batch
            nbatch = 10
            batch_size = int(np.ceil(num_length/nbatch))
            u_pred_test = []
            for ii in range(0, 10):
                if ii == 9:
                    x_vis_tensor = torch.tensor(x_vis[ii*batch_size:], 
                                    dtype=torch.float).to(device)
                else:
                    x_vis_tensor = torch.tensor(x_vis[ii*batch_size:(ii+1)*batch_size], 
                                    dtype=torch.float).to(device)
                
                u_pred_test_batch = to_numpy(model.forward(f_test_tensor[save_id:save_id+1, :], 
                                    x_pace_test_tensor[save_id:save_id+1, :, :], x_vis_tensor))
                if ii == 0:
                    u_pred_test = u_pred_test_batch
                else:
                    u_pred_test = np.concatenate((u_pred_test, u_pred_test_batch), axis=2)

            u_save = u_pred_test[0]*u_scale
            u_save = u_save.transpose()

            ## save
            dump_train = './vis/'+vis_case+'/LATs_pred'
            os.makedirs(dump_train, exist_ok=True)
            for ii in range(0, 7):
                fname= dump_train+"/LATs_nn_"+str(ii)+".dat"
                np.savetxt(fname, u_save[:, ii])

if __name__ == "__main__":
    main()
