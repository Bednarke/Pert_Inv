import mne
from mne.datasets import sample
import numpy as np
from sim_funcs import compute_fwds_stc_with_make_forward_dipole, compute_fwds_stc

local_data_path = 'C:\MEG\Local_mne_data'
data_path = sample.data_path()  # local copy of mne sample data
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
subjects_dir = data_path + '/subjects'
subject = 'sample'
trans = data_path + '\MEG\sample/sample_audvis_raw-trans.fif'
# Read files
cov = mne.read_cov(cov_fname)
info = mne.io.read_info(raw_fname)
########################################################################
# Setup our sources, bem
########################################################################
sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=None)  # Use this for all testing

perts = dict(imbalance=[], norm_error=[])
testsources = dict(rr=[], nn=[])
for i in range(0, 70):
    source = [0, 0, .001*i]
    normal = [.5, .5, 0]
    testsources['rr'].append(source)
    testsources['nn'].append(normal)

position = dict(rr=[0], nn=[0])
data = np.zeros((70, 22))
for i in range(0, 70):
    position['rr'][0] = testsources['rr'][i]
    position['nn'][0] = testsources['nn'][i]
    fwd_fixed, fwd_pert_fixed, stc = compute_fwds_stc(position, perts, sphere)  #just use coil dict
    evoked = mne.simulation.simulate_evoked(fwd_fixed, stc, info, cov, use_cps=True,
                                            iir_filter=None)
    evoked_pert = mne.simulation.simulate_evoked(fwd_pert_fixed, stc, info, cov, use_cps=True,
                                                 iir_filter=None)

    # evoked_pert.info = evoked.info
    dip_fit_long = mne.fit_dipole(evoked, cov_fname, sphere, trans)[0]
    dip_fit_pert = mne.fit_dipole(evoked_pert, cov_fname, sphere, trans)[0]
    '''
    xsum, ysum, zsum = 0, 0, 0
    for x in range(0, 1):
        dip_fit = mne.fit_dipole(evoked_pert, cov_fname, sphere, trans)[0]
        xsum += dip_fit.pos[0, 0]
        ysum += dip_fit.pos[0, 1]
        zsum += dip_fit.pos[0, 2]

    dip_fit_pert.pos[0, 0] = xsum / 5
    dip_fit_pert.pos[0, 1] = ysum / 5
   # dip_fit_pert.pos[0, 2] = zsum / 5
    '''
    del fwd_fixed, fwd_pert_fixed, evoked, evoked_pert
    for j in range(0, 3):
        data[i, 1+j] = testsources['rr'][i][j]
        data[i, 4 + j] = dip_fit_long.pos[0][j]
        data[i, 7 + j] = dip_fit_long.ori[0][j]
        data[i, 10 + j] = dip_fit_pert.pos[0][j]
        data[i, 13+j] = dip_fit_pert.ori[0][j]
        data[i, 20] = dip_fit_long.gof
        data[i, 21] = dip_fit_pert.gof


data_fname = local_data_path + '/10_percent_imbalance.csv'
np.savetxt(data_fname, data, delimiter=",")










