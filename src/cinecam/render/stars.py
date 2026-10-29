# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Star layer of the frames, in true colour: photometry, point spread function, stretch,
dots (the editor proxy look), automatic exposure.

Only numpy: used both by the render server (shamrock process) and by the editor.
"""

import numpy as np

from ..path.camera import ZNEAR, project

PC_M = 3.0857e16


def _cie_cmf(lam_nm):
    """CIE 1931 2-degree colour matching functions, multi-lobe gaussian fit (Wyman et al. 2013)."""

    def g(x, m, s1, s2):
        return np.exp(-0.5 * ((x - m) / np.where(x < m, s1, s2)) ** 2)

    x = 1.056 * g(lam_nm, 599.8, 37.9, 31.0) + 0.362 * g(lam_nm, 442.0, 16.0, 26.7) - 0.065 * g(lam_nm, 501.1, 20.4, 26.2)
    y = 0.821 * g(lam_nm, 568.8, 46.9, 40.5) + 0.286 * g(lam_nm, 530.9, 16.3, 31.1)
    z = 1.217 * g(lam_nm, 437.0, 11.8, 36.0) + 0.681 * g(lam_nm, 459.0, 26.0, 13.8)
    return x, y, z


# CIE XYZ -> linear sRGB (D65 white)
_XYZ_TO_RGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


class StarLight:
    """
    Stars as seen by a camera, in true colour.

    Photometry: stellar luminosity, radius and temperature from the sink mass (main
    sequence relations); the black body spectrum seen through the CIE colour matching
    functions gives linear sRGB fluxes (a lookup table in temperature); flux at the
    true distance from the camera, dimmed by the dust in the gas column between the
    camera and the star (Cardelli extinction at the sRGB channel wavelengths).

    Image: each visible star is stamped with an analytic point spread function (Moffat
    core, wavelength dependent width, Moffat halo, optional diffraction spikes) at its
    exact sub-pixel position, its stamp only as large as the part of it that will be
    visible. Or, without PSF, a round dot exactly as in the editor proxy (see dots).

    Tone mapping: one asinh stretch of the intensity (Lupton et al. 2004) applied to the
    three channels together, so the colours are kept; highlights fade to white.
    """

    sigma_sb = 5.670374419e-8
    Lsun = 3.828e26
    Rsun = 6.957e8

    # effective wavelengths of the sRGB channels [nm] and A_lambda / A_V there (Cardelli+1989, R_V=3.1)
    LAMBDA_RGB = np.array([610.0, 550.0, 465.0])
    EXT_RGB = np.array([0.890, 1.000, 1.234])
    # tau_V per Msun/pc^2 of gas: N_H = Sigma / (1.4 m_H), A_V = N_H / 1.87e21 cm^-2
    TAU_V_PER_SIGMA = 0.0439

    # stretch: intensity [W/m^2 per pixel] at which the asinh turns from linear to log, and
    # intensity mapped to 1 (white), at exposure 0
    I_SOFT = 10**-15.5
    I_WHITE = 10**-10.0
    REF_WIDTH = 1920.0  # PSF sizes below are in pixels of a 1920 px wide frame

    # PSF (reference pixels, at 550 nm): core and halo Moffat profiles, diffraction spikes
    CORE_FWHM = 1.6
    CORE_BETA = 3.0
    HALO_FWHM = 6.0
    HALO_BETA = 2.2
    HALO_FRAC = 0.06
    SPIKE_FRAC = 0.02
    RADIUS = 80  # stamp radius at most

    # dots (no PSF, and the editor proxy): radius DOT_R0 + DOT_R1 * brightness and opacity
    # DOT_A0 + DOT_A1 * brightness, in pixels of a DOT_REF_WIDTH wide frame (the viewfinder
    # at the editor's default size), drawn over the gas
    DOT_REF_WIDTH = 840.0
    DOT_R0, DOT_R1 = 0.7, 2.5
    DOT_A0, DOT_A1 = 0.35, 0.65

    def __init__(self):
        self._lut_T = np.logspace(np.log10(1500.0), np.log10(60000.0), 400)
        self._lut_rgb = self._colour_lut(self._lut_T)
        self._kernels = {}

    # -- photometry

    def mass_to_lum(self, m):
        cond = [m < 0.43, (m >= 0.43) & (m < 2.0), (m >= 2.0) & (m < 55.0), m >= 55]
        fns = [
            lambda m: self.Lsun * 0.23 * m**2.3,
            lambda m: self.Lsun * m**4,
            lambda m: self.Lsun * 1.4 * m**3.5,
            lambda m: self.Lsun * 32000.0 * m,
        ]
        return np.piecewise(m, cond, fns)

    def mass_to_radius(self, m):
        return self.Rsun * 0.85 * m**0.67

    def _colour_lut(self, T):
        """Linear sRGB per W of bolometric luminosity of a black body at T."""
        h, c, kb = 6.62607015e-34, 2.99792458e8, 1.380649e-23
        lam = np.linspace(380.0, 780.0, 401)
        cmf = np.array(_cie_cmf(lam))  # (3, nl)
        lm = lam * 1e-9
        with np.errstate(over="ignore"):
            B = 2 * h * c**2 / lm**5 / np.expm1(np.minimum(h * c / (np.outer(T, lm) * kb), 700.0))
        frac = np.pi * B / (self.sigma_sb * T[:, None] ** 4)  # spectral fraction of the luminosity, per m
        xyz = np.trapezoid(frac[:, None, :] * cmf[None], lm, axis=2)  # (nT, 3)
        return np.clip(xyz @ _XYZ_TO_RGB.T, 0.0, None)

    def luminosities(self, m_msun):
        """(N, 3) linear sRGB luminosities [W] and (N,) temperatures [K] from the masses."""
        m = np.maximum(np.asarray(m_msun, dtype=float), 1e-4)
        L = self.mass_to_lum(m)
        R = self.mass_to_radius(m)
        T = (L / (4 * np.pi * self.sigma_sb * R**2)) ** 0.25
        lT = np.log(np.clip(T, self._lut_T[0], self._lut_T[-1]))
        rgb = np.stack([np.interp(lT, np.log(self._lut_T), self._lut_rgb[:, k]) for k in range(3)], axis=1)
        return L[:, None] * rgb, T

    def colours(self, T):
        """Display colour (max channel 1) of black bodies at T, for markers."""
        lT = np.log(np.clip(T, self._lut_T[0], self._lut_T[-1]))
        rgb = np.stack([np.interp(lT, np.log(self._lut_T), self._lut_rgb[:, k]) for k in range(3)], axis=1)
        return rgb / rgb.max(axis=1, keepdims=True)

    def fluxes(self, state, pos, lum_rgb, sigma_los, pc_per_unit):
        """(N, 3) fluxes at the camera [W/m^2]."""
        d_m = np.maximum(np.linalg.norm(pos - state.pos, axis=1), 1e-6) * pc_per_unit * PC_M
        F = lum_rgb / (4 * np.pi * d_m**2)[:, None]
        if sigma_los is not None and state.extinction > 0:
            tau = self.TAU_V_PER_SIGMA * state.extinction * sigma_los[:, None] * self.EXT_RGB[None, :]
            F = F * np.exp(-tau)
        return F

    # -- stretch

    def stretch(self, img, exposure):
        """Linear RGB flux image -> display RGB in [0, 1], colours kept, highlights fade to white."""
        g = 10.0**exposure
        Is, Iw = self.I_SOFT, self.I_WHITE
        I = img.mean(axis=-1, keepdims=True) * g
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(I > 0, np.arcsinh(I / Is) / np.arcsinh(Iw / Is) / I, 0.0)
        out = img * g * k
        m = out.max(axis=-1, keepdims=True)
        over = np.maximum(m, 1.0)
        w = 1.0 - 1.0 / over  # how far above white: blend toward white
        return np.clip(out / over * (1 - w) + w, 0.0, 1.0)

    def visible_intensity(self, exposure):
        """Smallest pixel intensity that shows up (half a grey level)."""
        Is, Iw = self.I_SOFT, self.I_WHITE
        return Is * np.sinh(0.5 / 255 * np.arcsinh(Iw / Is)) / 10.0**exposure

    # -- point spread function

    @staticmethod
    def _moffat(r2, fwhm, beta):
        a2 = (fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))) ** 2
        return (beta - 1) / (np.pi * a2) * (1 + r2 / a2) ** -beta

    def _psf(self, x, y, s, lam, spikes, angle=0.0):
        """
        Analytic PSF (sum 1 over the plane) at offsets x, y [pixels, y down], size factor s,
        wavelength lam [nm]; spikes turned by angle [degrees, counterclockwise on screen].
        """
        w = s * lam / 550.0
        r2 = x * x + y * y
        core = self._moffat(r2, self.CORE_FWHM * w, self.CORE_BETA)
        halo = self._moffat(r2, self.HALO_FWHM * w, self.HALO_BETA)
        psf = (1 - self.HALO_FRAC - self.SPIKE_FRAC * spikes) * core + self.HALO_FRAC * halo
        if spikes:
            # 4 spikes (along the frame axes at angle 0): thin across, ~1/r^2 along, their
            # length grows with wavelength
            sw = 0.6 * self.CORE_FWHM * w
            L = 6.0 * w
            a = np.radians(angle)
            u = x * np.cos(a) - y * np.sin(a)  # along the first spike pair (y is down)
            v = x * np.sin(a) + y * np.cos(a)
            spk = 0.0
            for along, across in ((u, v), (v, u)):
                spk = spk + np.exp(-0.5 * (across / sw) ** 2) / (np.sqrt(2 * np.pi) * sw) / (1 + (along / L) ** 2) / (2 * np.pi * L)
            psf = psf + self.SPIKE_FRAC * spk
        return psf

    def kernels(self, nx, psf_scale, spikes, over=4, angle=0.0):
        """
        PSF stamps per sub-pixel phase: (over, over, 3, n, n) with n = 2R+1, each channel
        integrated over the pixels (4x4 sub-samples), plus the radial envelope used to size
        the stamps. Cached per resolution and look. angle: spike angle [degrees].
        """
        s = nx / self.REF_WIDTH * psf_scale
        angle = round(float(angle) % 90.0, 3) if spikes else 0.0  # 4 spikes: a quarter turn looks the same
        key = (round(s, 4), bool(spikes), over, angle)
        if key in self._kernels:
            return self._kernels[key]
        R = int(min(max(8, round(self.RADIUS * s)), 400))
        n = 2 * R + 1
        sub = (np.arange(4) + 0.5) / 4 - 0.5
        ii = np.arange(-R, R + 1, dtype=float)
        K = np.zeros((over, over, 3, n, n), np.float32)
        for py in range(over):
            for px in range(over):
                fy, fx = (py + 0.5) / over - 0.5, (px + 0.5) / over - 0.5  # star offset inside its pixel
                acc = np.zeros((3, n, n))
                for sy in sub:
                    for sx in sub:
                        Y = (ii + sy - fy)[:, None]
                        X = (ii + sx - fx)[None, :]
                        for c in range(3):
                            acc[c] += self._psf(X, Y, s, self.LAMBDA_RGB[c], spikes, angle)
                K[py, px] = acc / 16.0
        # smooth radial taper to zero at the stamp edge (no visible square cut on bright stars)
        rr = np.hypot(*np.meshgrid(ii, ii))
        K *= (np.clip(1 - (rr / (R + 0.5)) ** 2, 0, 1) ** 2).astype(np.float32)
        # resolution independent brightness: the same image at any width (sizes scale with it)
        K *= (nx / self.REF_WIDTH) ** 2
        env = np.array([K[:, :, :, rr >= r].max() if (rr >= r).any() else 0.0 for r in range(R + 1)])
        self._kernels[key] = (K, env, R)
        return self._kernels[key]

    # -- image

    def render(self, state, pos, lum_rgb, sigma_los, nx, ny, look, pc_per_unit=1.0, exposure=None):
        """
        Star layer for compose: with the PSF, float RGB in [0, 1] (screen-blended over the
        gas); without, premultiplied RGBA dots (drawn over the gas).
        """
        exposure = state.exposure if exposure is None else exposure
        img = np.zeros((ny, nx, 3 if look.get("star_psf", True) else 4))
        if len(pos) == 0:
            return img
        sx, sy, z = project(state, pos, nx, ny)
        margin = self.RADIUS * nx / self.REF_WIDTH * look.get("psf_scale", 1.0)
        ok = (z > ZNEAR) & (sx > -margin) & (sx < nx + margin) & (sy > -margin) & (sy < ny + margin)
        if not ok.any():
            return img
        F = self.fluxes(state, pos[ok], lum_rgb[ok], None if sigma_los is None else sigma_los[ok], pc_per_unit)
        sx, sy = sx[ok], sy[ok]
        if look.get("star_psf", True):
            lin = self._stamp(F, sx, sy, nx, ny, look, exposure)
            return self.stretch(lin, exposure)
        return self._dots(F, sx, sy, nx, ny, exposure)

    def _stamp(self, F, sx, sy, nx, ny, look, exposure, over=4):
        K, env, R = self.kernels(
            nx, look.get("psf_scale", 1.0), look.get("star_spikes", True), over, look.get("spike_angle", 0.0)
        )
        thr = self.visible_intensity(exposure)
        I = F.mean(axis=1)
        # stamp radius: as far as this star's PSF stays visible
        need = I[:, None] * env[None, :] >= thr
        rad = np.where(need.any(axis=1), R - np.argmax(need[:, ::-1], axis=1), -1)
        lin = np.zeros((ny + 2 * R, nx + 2 * R, 3))
        ix = np.floor(sx).astype(int)
        iy = np.floor(sy).astype(int)
        phx = np.clip(((sx - ix) * over).astype(int), 0, over - 1)
        phy = np.clip(((sy - iy) * over).astype(int), 0, over - 1)
        for k in np.flatnonzero(rad >= 0):
            r = int(rad[k])
            x0, y0 = ix[k] + R - r, iy[k] + R - r
            if x0 + 2 * r + 1 <= 0 or y0 + 2 * r + 1 <= 0 or x0 >= nx + 2 * R or y0 >= ny + 2 * R:
                continue
            st = K[phy[k], phx[k], :, R - r : R + r + 1, R - r : R + r + 1]  # (3, n, n)
            # clip the stamp to the padded canvas
            ax0, ay0 = max(x0, 0), max(y0, 0)
            ax1, ay1 = min(x0 + 2 * r + 1, nx + 2 * R), min(y0 + 2 * r + 1, ny + 2 * R)
            sub = st[:, ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0]
            lin[ay0:ay1, ax0:ax1] += np.moveaxis(sub, 0, -1) * F[k]
        return lin[R : R + ny, R : R + nx]

    def dots(self, F, exposure):
        """
        Dot look of stars with fluxes F (N, 3): brightness b in [0, 1] (asinh stretch of the
        PSF peak, as the PSF render), display colour (N, 3), radius [px of a DOT_REF_WIDTH
        frame], opacity. Shared by the render without PSF and the editor proxy.
        """
        I = F.mean(axis=1) * self._moffat(0.0, self.CORE_FWHM, self.CORE_BETA) * 10.0**exposure
        b = np.clip(np.arcsinh(I / self.I_SOFT) / np.arcsinh(self.I_WHITE / self.I_SOFT), 0.0, 1.0)
        col = np.clip(F / np.maximum(F.max(axis=1, keepdims=True), 1e-300), 0.0, 1.0) ** (1 / 2.2)
        return b, col, self.DOT_R0 + self.DOT_R1 * b, self.DOT_A0 + self.DOT_A1 * b

    def _dots(self, F, sx, sy, nx, ny, exposure, over=4):
        """Premultiplied RGBA layer of anti-aliased dots, the faint ones first (as the proxy)."""
        img = np.zeros((ny, nx, 4))
        b, col, rad, alpha = self.dots(F, exposure)
        rad = rad * nx / self.DOT_REF_WIDTH
        sub = (np.arange(over) + 0.5) / over
        for k in np.argsort(b, kind="stable"):
            if b[k] <= 0.5 / 255:
                continue
            r = rad[k]
            x0, x1 = max(int(np.floor(sx[k] - r)), 0), min(int(np.ceil(sx[k] + r)), nx)
            y0, y1 = max(int(np.floor(sy[k] - r)), 0), min(int(np.ceil(sy[k] + r)), ny)
            if x0 >= x1 or y0 >= y1:
                continue
            # pixel coverage of the disc, from over x over sub-samples per pixel
            yy = (np.arange(y0, y1)[:, None] + sub[None, :]).ravel()
            xx = (np.arange(x0, x1)[:, None] + sub[None, :]).ravel()
            inside = (yy[:, None] - sy[k]) ** 2 + (xx[None, :] - sx[k]) ** 2 <= r * r
            cov = inside.reshape(y1 - y0, over, x1 - x0, over).mean(axis=(1, 3))[..., None] * alpha[k]
            img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - cov) + np.append(col[k], 1.0) * cov
        return img

    # -- auto exposure

    def exposure_for(self, state, pos, lum_rgb, nx, ny, pc_per_unit, rank=10):
        """Exposure [dex] putting the rank-th brightest star in view just below white, None if no star."""
        if len(pos) == 0:
            return None
        sx, sy, z = project(state, pos, nx, ny)
        ok = (z > ZNEAR) & (sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny)
        if not ok.any():
            return None
        I = self.fluxes(state, pos[ok], lum_rgb[ok], None, pc_per_unit).mean(axis=1)
        I *= self._moffat(0.0, self.CORE_FWHM, self.CORE_BETA)  # peak pixel at the reference resolution
        v = np.sort(I)[::-1][min(rank, len(I)) - 1]
        return float(np.log10(0.5 * self.I_WHITE / v))


def star_exposure(path, t, starlight, nx, ny, window=1.0, samples=11):
    """
    Star exposure [dex] of the camera at time t: the key exposure, plus with
    look["auto_exposure"] an automatic part that keeps the brightest stars in view just
    below white. The automatic part is averaged over the frames within +-window/2 so it
    changes smoothly, and only depends on t (frames can be rendered in any order).
    """
    st = path.evaluate(t)
    if st is None:
        return 0.0
    tr = path.tracks
    if not path.look.get("auto_exposure", False) or tr is None:
        return st.exposure
    ts = np.clip(t + np.linspace(-window / 2, window / 2, samples), 0.0, path.duration)
    wts = np.exp(-0.5 * (np.linspace(-1, 1, samples) / 0.5) ** 2)
    vals, ws = [], []
    for ti, wi in zip(ts, wts):
        si = path.evaluate(ti)
        sk = tr.sinks_at(si.dump)
        if not len(sk["star_id"]):
            continue
        lum, _ = starlight.luminosities(sk["star_mass"])
        e = starlight.exposure_for(si, sk["star_pos"], lum, nx, ny, tr.pc_per_unit)
        if e is not None:
            vals.append(e)
            ws.append(wi)
    auto = float(np.clip(np.average(vals, weights=ws), -3.0, 3.0)) if vals else 0.0
    return st.exposure + auto

