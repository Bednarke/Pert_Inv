import mne
from mne.datasets import sample
from sim_funcs import compute_fwds_stc_with_make_forward_dipole, compute_fwds_stc

local_data_path = 'C:\MEG\Local_mne_data'
data_path = sample.data_path()  # local copy of mne sample data
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
# subjects_dir = data_path + '/subjects'
# subject = 'sample'
trans = data_path + '\MEG\sample/sample_audvis_raw-trans.fif'
# Read files
cov = mne.read_cov(cov_fname)
info = mne.io.read_info(raw_fname)
########################################################################
# Setup our sources, bem
########################################################################
sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=None)  # Use this for all testing
position = dict(rr=[[.05, .05, .05]], nn=[[0, 0, 1]])
times = [1]
coils = 'placeholder'
fwd_fixed, fwd_pert_fixed, stc = compute_fwds_stc(position, coils, sphere)

evoked = mne.simulation.simulate_evoked(fwd_fixed, stc, info, cov, use_cps=True,
                                        iir_filter=None)
evoked_pert = mne.simulation.simulate_evoked(fwd_pert_fixed, stc, info, cov, use_cps=True,
                                             iir_filter=None)

###############################################################################

dip_fit_long = mne.fit_dipole(evoked, cov_fname, sphere, trans)[0]
dip_fit_pert = mne.fit_dipole(evoked_pert, cov_fname, sphere, trans)[0]
print('Long fit, short fit:', dip_fit_long.pos, dip_fit_pert.pos)

#

