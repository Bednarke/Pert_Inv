import mne
from mne.datasets import sample
from sim_funcs import test
from _make_perturbed_forward import make_forward_solution
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
mri_head_t, trans = _get_trans(trans)
head_mri_t = invert_transform(mri_head_t)
info = mne.io.read_info(raw_fname)
########################################################################
# Setup our sources, bem
########################################################################
sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=None) # Use this for all testing

pos = dict(rr=[[.05, .05, .05]], nn=[[0, 0, 1]])
pos['rr'] = mne.transforms.apply_trans(head_mri_t, pos['rr'])  # invert back to mri
pos['nn'] = mne.transforms.apply_trans(head_mri_t, pos['nn'])
src = mne.setup_volume_source_space(subject=subject, pos=pos, mri=None,
                                    sphere=(0, 0, 0, 90), bem=None,
                                    surface=None, mindist=1.0, exclude=0.0,
                                    subjects_dir=None, volume_label=None,
                                    add_interpolator=True, verbose=None)
# print src
print('sources in MRI')
fwd = make_forward_solution(raw_fname, trans=trans, src=src, bem=sphere,
                            meg=True, eeg=False, mindist=1.0, n_jobs=1)
#  fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=sphere,
#                                meg=True, eeg=False, mindist=1.0, n_jobs=1)
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                         use_cps=True)
leadfield = fwd_fixed['sol']['data']
SourceLocations2 = fwd_fixed['src'][0]['rr']
print(SourceLocations2)
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
fixed_fwd_fname = data_path + '/MEG/sample/sample_audvis-sphere-2Source-fwd.fif'
mne.write_forward_solution(fixed_fwd_fname, fwd_fixed, overwrite=True, verbose=None)

###############################################################################
# IMPORTANT
# This is equivalent to the following code that explicitly applies the
# forward operator to a source estimate composed of the identity operator:
n_dipoles = leadfield.shape[1]
vertices = [src_hemi['vertno'] for src_hemi in fwd_fixed['src']]
stc = mne.VolSourceEstimate(1e-1 * np.eye(n_dipoles), vertices, tmin=0., tstep=1)
cov = mne.read_cov(cov_fname)
nave = 10000  # simulate average of 100 epochs
evoked = mne.simulation.simulate_evoked(fwd_fixed, stc, info, cov, nave=nave, use_cps=True,
                                        iir_filter=None)
###############################################################################
# Write Evoked
sim_ave_fname = local_data_path + '/MEG/sample/sample_audvis-sphere-2SourceEvoked-ave.fif'
mne.write_evokeds(sim_ave_fname, evoked)

###############################################################################
# Plot

###############################################################################
pos = dict(rr=[[.05, .05, .05]], nn=[[0, 0, 1]])
times = [1]
dip = mne.Dipole(times, pos['rr'], [1e-3], pos['nn'], [1], name=None, conf=None, khi2=None, nfree=None)
fwd_dip, stc_dip = mne.forward.make_forward_dipole(dip, sphere, evoked.info, trans)
fwd_dip_fixed = mne.convert_forward_solution(fwd_dip, surf_ori=True, force_fixed=True,
                                             use_cps=True)
leadfield_dip = fwd_dip_fixed['sol']['data']

###############################################################################
evoked_dip = mne.simulation.simulate_evoked(fwd_dip_fixed, stc_dip, evoked.info, cov, nave=nave, use_cps=True,
                                            iir_filter=None)
###############################################################################

'''
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

# print('Dipole fit at location', dip_fit.pos)
# dip_fit.plot_locations(trans, 'sample', subjects_dir, coord_frame='head', mode='orthoview')
dip_fit_long = mne.fit_dipole(evoked_dip, cov_fname, sphere, trans)[0]
dip_fit = mne.fit_dipole(evoked, cov_fname, sphere, trans)[0]
print('Long fit, short fit:', dip_fit_long.pos, dip_fit.pos)


