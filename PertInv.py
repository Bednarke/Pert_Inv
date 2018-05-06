import mne
from mne.datasets import sample
import numpy as np
from sim_funcs import fit_dips

local_data_path = 'C:\Users\/3l3ct\PycharmProjects\Pert_Inv\Local_mne_data'  # 'C:\Pert_Inv\Local_mne_data'

sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=None)  # Use this for all testing

perts = dict(mean_percent_imb=[5], variance_imb=[], preferred_side_imb=[0],
             mean_error_nn=[], variance_nn=[], preferred_direction_nn=[])
min_rad, max_rad = 0, 80
dip_fit_long, dip_fit_pert, testsources = fit_dips(min_rad, max_rad, sphere, perts)
data = np.zeros((max_rad - min_rad + 1, 22))
print(dip_fit_pert.gof)

for i in range(0, max_rad):
    for j in range(0, 3):
        data[i, 1+j] = testsources['rr'][i][j]
        data[i, 4 + j] = dip_fit_long.pos[i][j]
        data[i, 7 + j] = dip_fit_long.ori[i][j]
        data[i, 10 + j] = dip_fit_pert.pos[i][j]
        data[i, 13+j] = dip_fit_pert.ori[i][j]
        data[i, 20] = dip_fit_long.gof[i]
        data[i, 21] = dip_fit_pert.gof[i]


data_fname = local_data_path + '/%s_imb_%s_MaxRad_00z.csv' % (perts['mean_percent_imb'][0], max_rad)
np.savetxt(data_fname, data, delimiter=",")










