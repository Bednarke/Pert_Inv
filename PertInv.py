import mne
from mne.datasets import sample
import numpy as np
from sim_funcs import fit_dips


sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=None)  # Use this for all testing

perts = dict(max_percent_imb=10, variance_imb=0, preferred_side_imb=1,
             max_error_nn=0, variance_nn=0, preferred_direction_nn=0,
             max_translation=4)
sign = perts['preferred_side_imb']
max_imbalance = perts['max_percent_imb']
max_shift = perts['max_translation']
local_data_path = 'C:\Pert_Inv\Local_mne_data\side%s' % (sign,)
min_rad, max_rad = 0, 90

big_data = np.zeros((max_shift + 1, max_imbalance + 1, max_rad - min_rad + 1, 22))
nn = [0, 1, 1]
sourcenorm = [0, 1, 0] / np.linalg.norm([0, 1, 0])
for l in range(0, max_shift + 1):
    perts['max_translation'] = l
    for k in range(0, max_imbalance + 1):
        perts['max_percent_imb'] = k
        dip_fit_long, dip_fit_pert, testsources = fit_dips(min_rad, max_rad, nn, sphere, perts, sourcenorm)
        data = np.zeros((max_rad - min_rad + 1, 22))
        for i in range(0, max_rad):
            for j in range(0, 3):
                data[i, 1 + j] = round(testsources['rr'][i][j], 6)
                data[i, 4 + j] = round(dip_fit_long.pos[i][j], 6)
                data[i, 7 + j] = round(dip_fit_long.ori[i][j], 6)
                data[i, 10 + j] = round(dip_fit_pert.pos[i][j], 6)
                data[i, 13 + j] = round(dip_fit_pert.ori[i][j], 6)
                data[i, 19] = round(dip_fit_pert.pos[i][0], 6)
                data[i, 20] = round(dip_fit_long.gof[i], 6)
                data[i, 21] = round(dip_fit_pert.gof[i], 6)
        big_data[l, k] = data


        data_fname = local_data_path + '/%s_xshift_%s_imb_%s_sign_%s_MaxRad_00y_Norm%s%s%s.csv' \
                     % (l, k, sign, max_rad, nn[0], nn[1], nn[2])
        np.savetxt(data_fname, data, delimiter=",")
big_data_fname = local_data_path + '/%s_MaxXshift_%s_Max_imb%s_sign_%s_MaxRad_00y_Norm%s%s%s.csv'\
                 % (max_shift, max_imbalance, sign, max_rad, nn[0], nn[1], nn[2])
np.save(big_data_fname, big_data)









