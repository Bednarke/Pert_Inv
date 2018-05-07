import mne
from mne.datasets import sample
import numpy as np
from sim_funcs import fit_dips

local_data_path = 'C:\Pert_Inv\Local_mne_data'

sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=None)  # Use this for all testing

perts = dict(mean_percent_imb=0, variance_imb=[], preferred_side_imb=1,
             mean_error_nn=[], variance_nn=[], preferred_direction_nn=[])
min_rad, max_rad = 0, 90
max_imbalance = 5
specialname = 'xshift'
big_data = np.zeros((max_imbalance + 1, max_rad - min_rad + 1, 22))
nn = [1, 0, 1]
sign = perts['preferred_side_imb']
for k in range(0, max_imbalance + 1):
    perts['mean_percent_imb'] = k
    dip_fit_long, dip_fit_pert, testsources = fit_dips(min_rad, max_rad, nn, sphere, perts)
    data = np.zeros((max_rad - min_rad + 1, 22))
    for i in range(0, max_rad):
        for j in range(0, 3):
            data[i, 1 + j] = testsources['rr'][i][j]
            data[i, 4 + j] = dip_fit_long.pos[i][j]
            data[i, 7 + j] = dip_fit_long.ori[i][j]
            data[i, 10 + j] = dip_fit_pert.pos[i][j]
            data[i, 13 + j] = dip_fit_pert.ori[i][j]
            data[i, 20] = dip_fit_long.gof[i]
            data[i, 21] = dip_fit_pert.gof[i]
    big_data[k] = data
    data_fname = local_data_path + '/xshift_%s_imb%s_%s_MaxRad_00z_Norm%s%s%s.csv' \
                 % (perts['mean_percent_imb'], sign, max_rad, nn[0], nn[1], nn[2])
    np.savetxt(data_fname, data, delimiter=",")
big_data_fname = local_data_path + '/xshift_%s_Max_imb%s_%s_MaxRad_00z_Norm%s%s%s.csv'\
                 % (max_imbalance, sign, max_rad, nn[0], nn[1], nn[2])
np.save(big_data_fname, big_data)









