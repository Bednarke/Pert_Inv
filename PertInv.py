import mne
from mne.datasets import sample
import numpy as np
from _make_perturbed_forward import make_pert_forward_solution
from mne.transforms import (_ensure_trans, transform_surface_to, apply_trans,
                          _get_trans, invert_transform, _print_coord_trans, _coord_frame_name,
                          Transform)
from sim_funcs import compute_fwds_stc

local_data_path = 'C:\Users\/3l3ct\PycharmProjects\Pert_Inv\Local_mne_data'  # 'C:\Pert_Inv\Local_mne_data'
data_path = sample.data_path()  # local copy of mne sample data
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
subjects_dir = data_path + '/subjects'
subject = 'sample'
trans = data_path + '\MEG\sample/sample_audvis_raw-trans.fif'
# Read files
trans = data_path + '\MEG\sample/sample_audvis_raw-trans.fif'
mri_head_t, trans = _get_trans(trans)
head_mri_t = invert_transform(mri_head_t)

cov = mne.read_cov(cov_fname)
info = mne.io.read_info(raw_fname)
########################################################################
# Setup our sources, bem
########################################################################
sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=None)  # Use this for all testing

perts = dict(mean_percent_imb=[5], variance_imb=[], preferred_side_imb=[0],
             mean_error_nn=[], variance_nn=[], preferred_direction_nn=[])
testsources = dict(rr=[], nn=[])
min_rad = 0
max_rad = 1
nsources = max_rad - min_rad + 1
vertices = np.zeros((nsources, 1))
for i in range(min_rad, max_rad + 1):
    source = [0, 0, .001*i]
    normal = [.5, .5, 0]
    testsources['rr'].append(source)
    testsources['nn'].append(normal)
    vertices[i-min_rad] = i

pos = dict(rr=[0], nn=[0])
data = np.zeros((nsources, 22))
pos['rr'] = mne.transforms.apply_trans(head_mri_t, testsources['rr'])  # invert back to mri
pos['nn'] = mne.transforms.apply_trans(head_mri_t, testsources['nn'])
print(len(pos['rr']))

src = mne.setup_volume_source_space(subject=subject, pos=pos, mri=None,
                                    sphere=(0, 0, 0, 90), bem=None,
                                    surface=None, mindist=1.0, exclude=0.0,
                                    subjects_dir=None, volume_label=None,
                                    add_interpolator=True, verbose=None)
fwd_pert = make_pert_forward_solution(raw_fname, trans=trans, src=src, bem=sphere, perts=perts,
                                      meg=True, eeg=False, mindist=1.0, n_jobs=1)
fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=sphere,
                                meg=True, eeg=False, mindist=1.0, n_jobs=1)
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                         use_cps=True)
fwd_pert_fixed = mne.convert_forward_solution(fwd_pert, surf_ori=True, force_fixed=True,
                                              use_cps=True)

amplitude = 1e-5
source = np.eye(nsources)*amplitude


stc = mne.VolSourceEstimate(source, vertices, tmin=0., tstep=1)
evoked = mne.simulation.simulate_evoked(fwd_fixed, stc, info, cov, use_cps=True,
                                        iir_filter=None)
evoked_pert = mne.simulation.simulate_evoked(fwd_pert_fixed, stc, info, cov, use_cps=True,
                                             iir_filter=None)
dip_fit_long = mne.fit_dipole(evoked, cov_fname, sphere, trans)[0]
dip_fit_pert = mne.fit_dipole(evoked_pert, cov_fname, sphere, trans)[0]
print(dip_fit_pert.gof)

for i in range(0, max_rad):
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










