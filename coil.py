# -*- coding: utf-8 -*-


class Coil(object):
    u"""Coil class for containing geometry/calibration info

    Used to store coil positions, orientations, gradiometer imbalance, integration points.
    All data given in the sensor coordinate frame.

    Parameters
    ----------
    times : array, shape (n_dipoles,)
        The time instants at which each dipole was fitted (sec).
    pos : array, shape (n_dipoles, 3)
        The dipoles positions (m) in head coordinates.
    amplitude : array, shape (n_dipoles,)
        The amplitude of the dipoles (Am).
    ori : array, shape (n_dipoles, 3)
        The dipole orientations (normalized to unit length).
    gof : array, shape (n_dipoles,)
        The goodness of fit.
    name : str | None
        Name of the dipole.
    conf : dict
        Confidence limits in dipole orientation for "vol" in m^3 (volume),
        "depth" in m (along the depth axis), "long" in m (longitudinal axis),
        "trans" in m (transverse axis), "qlong" in Am, and "qtrans" in Am
        (currents). The current confidence limit in the depth direction is
        assumed to be zero (although it can be non-zero when a BEM is used).

        .. versionadded:: 0.15
    khi2 : array, shape (n_dipoles,)
        The χ^2 values for the fits.

        .. versionadded:: 0.15
    nfree : array, shape (n_dipoles,)
        The number of free parameters for each fit.

        .. versionadded:: 0.15

    See Also
    --------
    fit_dipole
    DipoleFixed
    read_dipole

    Notes
    -----
    This class is for sequential dipole fits, where the position
    changes as a function of time. For fixed dipole fits, where the
    position is fixed as a function of time, use :class:`mne.DipoleFixed`.
    """

    def __init__(self, times, pos, amplitude, ori, gof,
                 name=None, conf=None, khi2=None, nfree=None):  # noqa: D102
        self.times = np.array(times)
        self.pos = np.array(pos)
        self.amplitude = np.array(amplitude)
        self.ori = np.array(ori)
        self.gof = np.array(gof)
        self.name = name
        self.conf = deepcopy(conf) if conf is not None else dict()
        self.khi2 = np.array(khi2) if khi2 is not None else None
        self.nfree = np.array(nfree) if nfree is not None else None

    def __repr__(self):  # noqa: D105
        s = "n_times : %s" % len(self.times)
        s += ", tmin : %0.3f" % np.min(self.times)
        s += ", tmax : %0.3f" % np.max(self.times)
        return "<Dipole  |  %s>" % s

    def save(self, fname):
        """Save dipole in a .dip file.

        Parameters
        ----------
        fname : str
            The name of the .dip file.
        """
        # obligatory fields
        fmt = '  %7.1f %7.1f %8.2f %8.2f %8.2f %8.3f %8.3f %8.3f %8.3f %6.2f'
        header = ('#   begin     end   X (mm)   Y (mm)   Z (mm)'
                  '   Q(nAm)  Qx(nAm)  Qy(nAm)  Qz(nAm)    g/%')
        t = self.times[:, np.newaxis] * 1000.
        gof = self.gof[:, np.newaxis]
        amp = 1e9 * self.amplitude[:, np.newaxis]
        out = (t, t, self.pos / 1e-3, amp, self.ori * amp, gof)

        # optional fields
        fmts = dict(khi2=('    khi^2', ' %8.1f', 1.),
                    nfree=('  free', ' %5d', 1),
                    vol=('  vol/mm^3', ' %9.3f', 1e9),
                    depth=('  depth/mm', ' %9.3f', 1e3),
                    long=('  long/mm', ' %8.3f', 1e3),
                    trans=('  trans/mm', ' %9.3f', 1e3),
                    qlong=('  Qlong/nAm', ' %10.3f', 1e9),
                    qtrans=('  Qtrans/nAm', ' %11.3f', 1e9),
                    )
        for key in ('khi2', 'nfree'):
            data = getattr(self, key)
            if data is not None:
                header += fmts[key][0]
                fmt += fmts[key][1]
                out += (data[:, np.newaxis] * fmts[key][2],)
        for key in ('vol', 'depth', 'long', 'trans', 'qlong', 'qtrans'):
            data = self.conf.get(key)
            if data is not None:
                header += fmts[key][0]
                fmt += fmts[key][1]
                out += (data[:, np.newaxis] * fmts[key][2],)
        out = np.concatenate(out, axis=-1)

        # NB CoordinateSystem is hard-coded as Head here
        with open(fname, 'wb') as fid:
            fid.write('# CoordinateSystem "Head"\n'.encode('utf-8'))
            fid.write((header + '\n').encode('utf-8'))
            np.savetxt(fid, out, fmt=fmt)
            if self.name is not None:
                fid.write(('## Name "%s dipoles" Style "Dipoles"'
                           % self.name).encode('utf-8'))


