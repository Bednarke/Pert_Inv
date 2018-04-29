import mne
from mne.datasets import sample
from sim_funcs import *
import numpy as np  # noqa
from mne.transforms import (_ensure_trans, transform_surface_to, apply_trans,
                          _get_trans, invert_transform, _print_coord_trans, _coord_frame_name,
                          Transform)
local_data_path = 'C:\MEG\Local_mne_data'
data_path = sample.data_path()  # local copy of mne sample data
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
subjects_dir = data_path + '/subjects'
subject = 'sample'
trans = data_path + '\MEG\sample/sample_audvis_raw-trans.fif'
# Read files
mri_head_t, trans = _get_trans(trans)
head_mri_t = invert_transform(mri_head_t)
cov = mne.read_cov(cov_fname)
info = mne.io.read_info(raw_fname)
########################################################################
# Setup our sources, bem
########################################################################
sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=None) # Use this for all testing
position = dict(rr=[[.05, .05, .05]], nn=[[0, 0, 1]])
times = [1]
coils = 'placeholder'
fwd_fixed, fwd_pert_fixed, stc = compute_fwds_stc(position, coils, sphere)
evoked = mne.simulation.simulate_evoked(fwd_fixed, stc, info, cov, use_cps=True,
                                        iir_filter=None)
evoked_pert = mne.simulation.simulate_evoked(fwd_pert_fixed, stc, info, cov, use_cps=True,
                                             iir_filter=None)
###############################################################################
dip = mne.Dipole(times, position['rr'], [1e-5], position['nn'], [1],
                 name='index', conf=None, khi2=None, nfree=None)
fwd_dip, stc_dip = mne.forward.make_forward_dipole(dip, sphere, evoked.info, trans)
fwd_dip_fixed = mne.convert_forward_solution(fwd_dip, surf_ori=True, force_fixed=True,
                                             use_cps=True)
leadfield_dip = fwd_dip_fixed['sol']['data']

###############################################################################
evoked_dip = mne.simulation.simulate_evoked(fwd_dip_fixed, stc_dip, evoked.info, cov, use_cps=True,
                                            iir_filter=None)
###############################################################################

'''
CHANGE POS TO POSITION
xsum, ysum, zsum = 0, 0, 0
for x in xrange(10):
    dip_fit = mne.fit_dipole(evoked_dip, cov_fname, sphere, trans)[0]
    xsum += dip_fit.pos[0, 0]
    ysum += dip_fit.pos[0, 1]
    zsum += dip_fit.pos[0, 2]

dip_fit.pos[0, 0] = xsum / 10
dip_fit.pos[0, 1] = ysum / 10
dip_fit.pos[0, 2] = zsum / 10
'''
# dip_fit = mne.fit_dipole(evoked, cov_fname, sphere, trans)[0]
# Plot the result in 3D brain with the MRI image.

# print('Dipole fit at location', dip_fit.position)
# dip_fit.plot_locations(trans, 'sample', subjects_dir, coord_frame='head', mode='orthoview')
dip_fit_long = mne.fit_dipole(evoked, cov_fname, sphere, trans)[0]
dip_fit_pert = mne.fit_dipole(evoked_pert, cov_fname, sphere, trans)[0]
dip_fit = mne.fit_dipole(evoked_dip, cov_fname, sphere, trans)[0]
print('Long fit, short fit:', dip_fit_long.pos, dip_fit_pert.pos)


