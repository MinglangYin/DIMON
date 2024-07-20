import numpy as np
import os

case_ids = [696, 706]

for id in case_ids:
    os.makedirs("./HCM_new_"+str(id)+"/LATs_err", exist_ok=True)

    for i in range(0, 7):
        nn_pred = np.loadtxt("./HCM_new_"+str(id)+"/LATs_pred/LATs_nn_"+str(i)+".dat")
        fem = np.loadtxt("./HCM_new_"+str(id)+"/LATs/LATs_"+str(i)+".dat")
        err = abs(nn_pred - fem)
        np.savetxt("./HCM_new_"+str(id)+"/LATs_err/LATs_err_"+str(i)+".dat", err)
