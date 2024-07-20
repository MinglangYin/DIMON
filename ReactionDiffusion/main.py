"""
Author: Minglang Yin, minglang_yin@brown.edu
FitzHugh-Nagumo Equation. 
Details see "Fenics/FitzHugh-Nagumo/anisotropic_grf"

Branch Net 1 Input:       [L1, L2, ..., Lm],
Branch Net 2 Input:       [g1, g2, ..., gk]
Output:                 [u]
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
import pickle as pickle

def main():
    ## hyperparameters
    args = ParseArgument()
    epochs = args.epochs
    device = args.device
    save_step = args.save_step
    test_model = args.test_model
    restart = args.restart
    if test_model == 1:
        load_path = "./CheckPts/model_chkpts.pt"
        test_path = "./RD_test.mat"
    if restart == 1:
        load_path = "./CheckPts/model_chkpts.pt"


    num_umode = 50
    num_geomode = 4 #20
    if_normal = 0

    dim_br_u = [num_umode, 300, 300, 200]
    dim_br_geo = [num_geomode, 200, 200, 200]
    dim_tr = [3, 200, 200, 200]

    num_train = 2700
    num_test = 300
    num_cases = num_train + num_test

    ## create folders
    dump_train = './Predictions/Train/'
    dump_test = './Predictions/Test/'
    model_path = 'CheckPts/model_chkpts.pt'
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)
    
    ## dataset
    # datafile = "../RD_data.mat"
    # dataset = io.loadmat(datafile)
    datafile = "../dataset_pod_new.npy"
    dataset = np.load(datafile, allow_pickle=True)

    nskip = 2
    u_data = dataset['u_data'][:, ::nskip]

    x_data = dataset['x_uni']
    x_mesh = dataset["x_mesh_data"]
    # f_u = dataset["u_init"]
    f_u = dataset["coeff_u0"]
    f_g = dataset["param_data"]
    # f_g = dataset["coeff_geo"]
   

    ###############################
    ## expand x with time
    t = dataset["t"]
    t_num = t.shape[1]
    tx_ext = np.zeros((t_num, x_data.shape[0], 3))
    for i in range(0, t_num):
        tx_ext[i, :, 0] = t[0, i]
        tx_ext[i, :, 1:3] = x_data
    tx_ext = tx_ext[::nskip]
    num_snap = tx_ext.shape[0]
    num_pts = x_data.shape[0]
    tx_ext = tx_ext.reshape(-1, 3)
    ###############################

    ##########################
    ## testing/training dataset
    u_train = u_data[:num_train] # start from snapshot 1
    f_u_train = f_u[:num_train] # predict based on previous snapshot
    f_g_train = f_g[:num_train]

    u_test = u_data[num_train:(num_train+num_test)]    
    f_u_test = f_u[num_train:(num_train+num_test)]
    f_g_test = f_g[num_train:(num_train+num_test)]


    ## normalization
    if if_normal == 1:
        f_u_mean = f_u_train.mean(axis=0)
        f_u_std = f_u_train.std(axis=0)
        
        np.savetxt("./f_u_mean_std.txt", np.concatenate((f_u_mean[:, None], f_u_std[:, None]), axis=1))
        
        f_u_train = (f_u_train - f_u_mean)/f_u_std
        f_u_test = (f_u_test - f_u_mean)/f_u_std

    ## tensor
    u_train_tensor = torch.tensor(u_train, dtype=torch.float).to(device)
    f_u_train_tensor = torch.tensor(f_u_train, dtype=torch.float).to(device)
    f_g_train_tensor = torch.tensor(f_g_train, dtype=torch.float).to(device)

    # u_test_tensor = torch.tensor(u_test, dtype=torch.float).to(device)
    f_u_test_tensor = torch.tensor(f_u_test, dtype=torch.float).to(device)
    f_g_test_tensor = torch.tensor(f_g_test, dtype=torch.float).to(device)
    xt_tensor = torch.tensor(tx_ext, dtype=torch.float).to(device)

    ## initialization
    model = opnn(dim_br_u, dim_br_geo, dim_tr, num_snap, num_pts).to(device) ## note this is not POD-OPNN. Use POD mode to express functions
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002)
    
    if test_model == 0:
        train_loss = np.zeros((args.epochs, 1))
        test_loss = np.zeros((args.epochs, 1))

        if restart == 1:
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['model_state_dict'])

        ## training
        def train(epoch, f_u, f_g, x, y):
            model.train()
            def closure():
                optimizer.zero_grad()
                y_pred = model.forward(f_u, f_g, x)

                loss = ((y_pred - y)**2).mean()
                train_loss[epoch, 0] = loss
                loss.backward()
                return loss
            optimizer.step(closure)

        ## Iterations
        print('start training...', flush=True)
        tic = time.time()
        for epoch in range(0, epochs):
            if epoch == 50000:
                optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
            elif epoch == 90000:
                # optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
                optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
            # elif epoch == epochs - 1000:
            #     optimizer = torch.optim.LBFGS(model.parameters())

            ## Training
            train(epoch, f_u_train_tensor, f_g_train_tensor, xt_tensor, u_train_tensor)
            ## Testing
            y_pred = to_numpy(model.forward(f_u_test_tensor, f_g_test_tensor, xt_tensor))
            loss_tmp = ((y_pred - u_test)**2).mean() # denote u_test as testing cardiac field
            test_loss[epoch, 0] = loss_tmp

            ## testing error
            if epoch%100 == 0:
                print(f'Epoch: {epoch}, Train Loss: {train_loss[epoch, 0]:.6f}, Test Loss: {test_loss[epoch, 0]:.6f}', flush=True)
                
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

        ## save train
        u_train_pred = model.forward(f_u_train_tensor, f_g_train_tensor, xt_tensor)
        u_pred = to_numpy(u_train_pred)
        u_true = u_train
        num_t = u_train.shape[1]


        for i in range(0, 10):
            u_pred_plot = u_pred[i, :]
            u_true_plot = u_true[i, :]
            os.makedirs(dump_train + str(i), exist_ok=True)
            print(f"printing training case: {i}")
            
            for tt in range(0, num_t):
                fig = plt.figure(constrained_layout=False, figsize=(15, 5))
                gs = fig.add_gridspec(1, 3)

                ax = fig.add_subplot(gs[0])
                # h = ax.imshow(u_true_plot[tt], label="True", vmin=-0.75, vmax=1)
                h = ax.scatter(x_mesh[i, :, 0], x_mesh[i, :, 1], c =u_true_plot[tt])
                ax.set_aspect(1)
                plt.colorbar(h)
                ax.set_title("True data")

                ax = fig.add_subplot(gs[1])
                # h = ax.imshow(u_pred_plot[tt], label="Pred", vmin=-0.75, vmax=1)
                h = ax.scatter(x_mesh[i, :, 0], x_mesh[i, :, 1], c=u_pred_plot[tt])
                ax.set_aspect(1)
                plt.colorbar(h)
                ax.set_title("prediction")

                ax = fig.add_subplot(gs[2])
                # h = ax.imshow(u_pred_plot[tt] - u_true_plot[tt], label="Pred")
                h = ax.scatter(x_mesh[i, :, 0], x_mesh[i, :, 1], c=u_pred_plot[tt] - u_true_plot[tt])
                ax.set_aspect(1)
                plt.colorbar(h)
                ax.set_title("err")

                fig.savefig(dump_train + str(i) + '/snapshot_' + str(tt) + '.png')
                plt.close()
        
        ## save test
        u_test_pred = model.forward(f_u_test_tensor, f_g_test_tensor, xt_tensor)
        u_pred = to_numpy(u_test_pred)
        u_true = u_test

        for i in range(0, 10):
            u_pred_plot = u_pred[i, :]
            u_true_plot = u_true[i, :]
            os.makedirs(dump_test + str(i), exist_ok=True)
            print(f"printing testing case: {i}")

            for tt in range(0, num_t):
                fig = plt.figure(constrained_layout=False, figsize=(15, 5))
                gs = fig.add_gridspec(1, 3)

                ## 
                ax = fig.add_subplot(gs[0])
                # h = ax.imshow(u_true_plot[tt], label="True", vmin=-0.75, vmax=1)
                h = ax.scatter(x_mesh[i, :, 0], x_mesh[i, :, 1], c=u_true_plot[tt])
                ax.set_aspect(1)
                plt.colorbar(h)
                ax.set_title("True data")

                ax = fig.add_subplot(gs[1])
                # h = ax.imshow(u_pred_plot[tt], label="Pred", vmin=-0.75, vmax=1)
                h = ax.scatter(x_mesh[i, :, 0], x_mesh[i, :, 1], c =u_pred_plot[tt])
                ax.set_aspect(1)
                plt.colorbar(h)
                ax.set_title("prediction")

                ax = fig.add_subplot(gs[2])
                # h = ax.imshow(u_pred_plot[tt] - u_true_plot[tt], label="Pred")
                h = ax.scatter(x_mesh[i, :, 0], x_mesh[i, :, 1], c=u_pred_plot[tt] - u_true_plot[tt])
                ax.set_aspect(1)
                plt.colorbar(h)
                ax.set_title("err")

                fig.savefig(dump_test + str(i) + '/snapshot_' + str(tt) + '.png')
                plt.close()
    else:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        ## save 
        u_test_pred = model.forward(f_u_test_tensor, f_g_test_tensor, xt_tensor)
        u_pred = to_numpy(u_test_pred)
        u_true = u_test
        abs_err = (abs(u_pred - u_true)).mean(axis=2)
        rel_l2_err = (np.linalg.norm(u_true - u_pred, axis=2)/np.linalg.norm(u_true, axis=2)) #.reshape(num_test, 26)
        
        np.savetxt("abs_err.txt", abs_err)
        np.savetxt("rel_l2_err.txt", rel_l2_err)
        exit()


        ##
        dataset_test = io.loadmat(test_path)

        ##############
        ## save test
        u_test_pred = model.forward(f_u_test_tensor, f_g_test_tensor, xt_tensor)
        u_pred = to_numpy(u_test_pred)


        u_true = u_test
        u_pred_save = []
        err_pred_save = []

        for i in [1, 2, 3]: 
            u_pred_plot = u_pred[i, :]
            u_true_plot = u_true[i, :]
            # os.makedirs(dump_test + str(i), exist_ok=True)
            print(f"printing testing case: {i}")

            for tt in range(0, 26): #### need to adjust by case
                u_pred_plot = u_pred[i]
                u_true_plot = u_true[i]
                x_coord = x_mesh[i+num_train]

                # print(u_true_plot.shape, x_coord.shape, u_pred_plot.shape)
                # exit()

                
                # print(f"printing testing case: {i}")

                # print(u_true_plot.shape, u_true_plot[tt].shape, tt)
                # exit()

                # ###
                fig = plt.figure(constrained_layout=False, figsize=(15, 5))
                gs = fig.add_gridspec(1, 3)

                ax = fig.add_subplot(gs[0])
                h = ax.scatter(x_coord[:, 0], x_coord[:, 1], c=u_true_plot[tt])
                plt.colorbar(h)
                ax.set_aspect(1)
                ax.set_title("True data")

                ax = fig.add_subplot(gs[1])
                h = ax.scatter(x_coord[:, 0], x_coord[:, 1], c=u_pred_plot[tt])
                plt.colorbar(h)
                ax.set_aspect(1)
                ax.set_title("prediction")

                ax = fig.add_subplot(gs[2])
                h = ax.scatter(x_coord[:, 0], x_coord[:, 1], c=u_true_plot[tt] - u_pred_plot[tt])
                plt.colorbar(h)
                ax.set_aspect(1)
                ax.set_title("err")

                fig.savefig(dump_test + "case_" + str(i) + "_"+ str(tt) +"_img.png")
                plt.close()

            u_pred_save.append(u_pred_plot)
            err_pred_save.append(abs(u_pred_plot - u_true_plot))


        save_lib = {}
        save_lib["u_pred_canonical"] = u_pred_save
        save_lib["x_canonical"] = x_coord
        save_lib["x_mesh"] = dataset_test["x_mesh_data"]
        save_lib["mesh_coords"] = dataset_test["x_mesh_coords_fine"]
        save_lib["mesh_cells"] = dataset_test["x_mesh_cells_fine"]
        save_lib["err_pred_save"] = err_pred_save
        io.savemat("rd_model_pred_vis.mat", save_lib)
    

if __name__ == "__main__":
    main()
