# House funcs here so PertInv is just scripting
import mne
from _make_perturbed_forward import make_pert_forward_solution, make_pert_forward_dipole
from mne.datasets import sample
import numpy as np  # noqa
from mne.transforms import (_ensure_trans, transform_surface_to, apply_trans,
                          _get_trans, invert_transform, _print_coord_trans, _coord_frame_name,
                          Transform)
# local_data_path = 'C:\MEG\Local_mne_data'
data_path = sample.data_path()  # local copy of mne sample data
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
subjects_dir = data_path + '/subjects'
subject = 'sample'
trans = data_path + '\MEG\sample/sample_audvis_raw-trans.fif'
mri_head_t, trans = _get_trans(trans)
head_mri_t = invert_transform(mri_head_t)


def compute_fwds_stc(position, coils, sphere):
    pos = position.copy()
    pos['rr'] = mne.transforms.apply_trans(head_mri_t, position['rr'])  # invert back to mri
    pos['nn'] = mne.transforms.apply_trans(head_mri_t, position['nn'])
    src = mne.setup_volume_source_space(subject=subject, pos=pos, mri=None,
                                        sphere=(0, 0, 0, 90), bem=None,
                                        surface=None, mindist=1.0, exclude=0.0,
                                        subjects_dir=None, volume_label=None,
                                        add_interpolator=True, verbose=None)
    fwd_pert = make_pert_forward_solution(raw_fname, trans=trans, src=src, bem=sphere,
                                          meg=True, eeg=False, mindist=1.0, n_jobs=1)
    fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=sphere,
                                    meg=True, eeg=False, mindist=1.0, n_jobs=1)
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                             use_cps=True)
    fwd_pert_fixed = mne.convert_forward_solution(fwd_pert, surf_ori=True, force_fixed=True,
                                                  use_cps=True)

    amplitude = 1e-5
    stc = mne.VolSourceEstimate(amplitude * np.eye(1), [[0]], tmin=0., tstep=1)
    return fwd_fixed, fwd_pert_fixed, stc


def compute_fwds_stc_with_make_forward_dipole(dip, info, coils, sphere):
    dipole = dip.copy()
    fwd, stc = mne.forward.make_forward_dipole(dipole, sphere, info, trans)
    fwd_pert, stc = make_pert_forward_dipole(dipole, sphere, info, trans)
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                             use_cps=True)

    fwd_pert_fixed = mne.convert_forward_solution(fwd_pert, surf_ori=True, force_fixed=True,
                                                  use_cps=True)
    return fwd_fixed, fwd_pert_fixed, stc





