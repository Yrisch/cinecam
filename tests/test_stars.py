# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""Star layer: true colour photometry, PSF, stretch, dots, auto exposure (render/stars.py)."""

import numpy as np
import pytest
from conftest import sink_pos

from cinecam.path.camera import CamState
from cinecam.path.campath import CameraPath, Keyframe
from cinecam.render.stars import StarLight, star_exposure

SL = StarLight()
LOOK = {"psf_scale": 1.0, "star_psf": True, "star_spikes": True}


def view(**kw):
    return CamState(pos=[0, -10, 0], target=[0, 0, 0], up=[0, 0, 1], **kw)


def test_colours_follow_the_temperature():
    red, sun, blue = SL.colours(np.array([3000.0, 5800.0, 30000.0]))
    assert red[0] == 1 and red[2] < 0.3  # cool stars: orange-red
    assert sun.min() > 0.75  # the Sun: close to white
    assert blue[2] == 1 and blue[0] < 0.5  # hot stars: blue


def test_luminosity_and_temperature_grow_with_mass():
    lum, T = SL.luminosities(np.array([0.2, 1.0, 5.0, 40.0]))
    assert lum.shape == (4, 3)
    assert (np.diff(lum.sum(axis=1)) > 0).all() and (np.diff(T) > 0).all()
    assert T[1] == pytest.approx(6261, rel=0.01)


def test_fluxes_inverse_square_and_reddening():
    lum, _ = SL.luminosities(np.array([5.0, 5.0]))
    pos = np.array([[0, 0, 0], [0, 10, 0]])  # 10 and 20 pc from the camera
    F = SL.fluxes(view(), pos, lum, None, 1.0)
    np.testing.assert_allclose(F[0] / F[1], 4.0)
    Fd = SL.fluxes(view(), pos[:1], lum[:1], np.array([100.0]), 1.0)  # behind 100 Msun/pc^2 of gas
    ratio = Fd[0] / F[0]
    assert ratio[2] < ratio[1] < ratio[0] < 1  # blue dimmed most: reddening
    np.testing.assert_allclose(SL.fluxes(view(extinction=0.0), pos[:1], lum[:1], np.array([100.0]), 1.0), F[:1])


def test_stretch_keeps_colours_and_fades_to_white():
    faint = np.array([[[2e-15, 1e-15, 0.5e-15]]])
    out = SL.stretch(faint, 0.0)[0, 0]
    np.testing.assert_allclose(out / out[0], [1.0, 0.5, 0.25], rtol=1e-9)  # hue kept
    bright, brighter = SL.stretch(faint * 1e3, 0.0)[0, 0], SL.stretch(faint * 1e8, 0.0)[0, 0]
    assert bright.max() == brighter.max() == 1.0
    assert brighter.min() > bright.min() > out.min()  # highlights fade toward white
    assert (SL.stretch(faint, 1.0) > SL.stretch(faint, 0.0)).all()  # exposure brightens


def test_psf_kernels_are_normalised_and_resolution_independent():
    K, env, R = SL.kernels(1920, 1.0, True)
    assert R == SL.RADIUS
    np.testing.assert_allclose(K[0, 0].sum(axis=(1, 2)), 1.0, atol=0.01)  # unit PSF (the taper loses < 1%)
    K2, _, R2 = SL.kernels(960, 1.0, True)
    assert R2 == R // 2
    np.testing.assert_allclose(K2[0, 0].sum(axis=(1, 2)), 0.25, atol=0.003)  # same pixel values at half size


def test_spikes_along_the_frame_axes():
    K, _, R = SL.kernels(1920, 1.0, True)
    k = K[2, 2, 1]  # green, star centred
    assert k[R, R + 30] > 10 * k[R + 21, R + 21]  # on the axis vs on the diagonal at the same distance
    K0, _, _ = SL.kernels(1920, 1.0, False)
    assert K0[2, 2, 1][R, R + 30] < k[R, R + 30] / 10


def test_spike_angle_turns_the_spikes_counterclockwise():
    K, _, R = SL.kernels(1920, 1.0, True, angle=30.0)
    k = K[2, 2, 1]
    a = np.radians(30.0)
    on = k[R - round(30 * np.sin(a)), R + round(30 * np.cos(a))]  # 30 px up-right at 30 degrees (rows go down)
    assert on > 10 * k[R, R + 30]  # no longer along the frame axis
    assert on > 10 * k[R + round(30 * np.sin(a)), R + round(30 * np.cos(a))]  # not clockwise
    np.testing.assert_array_equal(SL.kernels(1920, 1.0, True, angle=90.0)[0], SL.kernels(1920, 1.0, True)[0])
    np.testing.assert_array_equal(SL.kernels(1920, 1.0, False, angle=30.0)[0], SL.kernels(1920, 1.0, False)[0])


def test_render_psf():
    pos = np.array([[0.0, 0.0, 0.0]])
    lum, _ = SL.luminosities(np.array([5.0]))
    img = SL.render(view(), pos, lum, None, 384, 216, LOOK)
    assert img.shape == (216, 384, 3) and 0 <= img.min() and img.max() <= 1
    assert img[108, 192].max() > 0.5  # star in the centre
    assert SL.render(view(), pos, lum, None, 384, 216, LOOK, exposure=-3).max() < img.max()


def test_render_without_psf_draws_the_proxy_dots():
    """No PSF: an RGBA dot of the colour, radius and opacity of the editor proxy (StarLight.dots)."""
    pos = np.array([[0.0, 0.0, 0.0]])
    lum, _ = SL.luminosities(np.array([5.0]))
    nx, ny = 1680, 945  # twice the dot reference width
    img = SL.render(view(), pos, lum, None, nx, ny, dict(LOOK, star_psf=False))
    assert img.shape == (ny, nx, 4)
    b, col, rad, alpha = SL.dots(SL.fluxes(view(), pos, lum, None, 1.0), 0.0)
    cy, cx = ny // 2, nx // 2  # the star sits on a pixel corner: the 4 pixels around are inside the dot
    np.testing.assert_allclose(img[cy, cx, 3], alpha[0])
    np.testing.assert_allclose(img[cy, cx, :3], col[0] * alpha[0])  # premultiplied
    covered = (img[..., 3] > 0).sum()
    assert covered == pytest.approx(np.pi * (2 * rad[0]) ** 2, rel=0.2)  # radius x 2 at twice the width
    psf = SL.render(view(), pos, lum, None, nx, ny, LOOK)
    assert covered < (psf.max(axis=2) > 0.01).sum()  # no glow


def test_dots_brightness_matches_the_psf_stretch():
    """A dot is as bright as the peak pixel of the PSF render of the same star."""
    lum, _ = SL.luminosities(np.array([0.3, 2.0, 20.0]))
    F = SL.fluxes(view(), np.array([[0, 0, 0], [0, 0, 0], [0, 40, 0]]), lum, None, 1.0)
    b, col, rad, alpha = SL.dots(F, -4.0)  # dark exposure: none saturated
    assert b.max() < 1 and (np.diff(b) > 0).all() and (np.diff(rad) > 0).all() and (np.diff(alpha) > 0).all()
    np.testing.assert_allclose(col.max(axis=1), 1.0)
    grey_peak = np.repeat(F.mean(axis=1, keepdims=True), 3, axis=1) * SL._moffat(0.0, SL.CORE_FWHM, SL.CORE_BETA)
    np.testing.assert_allclose(SL.stretch(grey_peak[:, None, :], -4.0)[:, 0, 0], b)


def test_star_behind_the_camera_is_not_drawn():
    lum, _ = SL.luminosities(np.array([5.0]))
    assert SL.render(view(), np.array([[0.0, -20.0, 0.0]]), lum, None, 64, 36, LOOK).max() == 0


def test_exposure_for_puts_the_rank_th_star_below_white():
    lum, _ = SL.luminosities(np.full(3, 2.0))
    pos = np.array([[0, 0, 0], [1, 5, 0], [-1, 10, 0]], float)
    e = SL.exposure_for(view(), pos, lum, 384, 216, 1.0, rank=2)
    I = SL.fluxes(view(), pos[1:2], lum[1:2], None, 1.0).mean() * SL._moffat(0.0, SL.CORE_FWHM, SL.CORE_BETA)
    assert I * 10**e == pytest.approx(0.5 * SL.I_WHITE)
    assert SL.exposure_for(view(), np.array([[0, -20.0, 0]]), lum[:1], 384, 216, 1.0) is None


def test_star_exposure_auto_is_smooth_bounded_and_order_independent(tracks):
    keys = [Keyframe(0, CamState(pos=[0, -3, 0], target=sink_pos(0, 0), up=[0, 0, 1], exposure=0.5)),
            Keyframe(2, CamState(pos=[0, -0.5, 0], target=sink_pos(0, 2), up=[0, 0, 1], dump=4, exposure=0.5))]  # fmt: skip
    p = CameraPath(keyframes=keys, duration=2, dumps=["d"] * tracks.ndump)
    p.set_tracks(tracks)
    assert star_exposure(p, 1.0, SL, 384, 216) == 0.5  # auto exposure off: the key exposure
    p.look["auto_exposure"] = True
    ts = np.linspace(0, 2, 9)
    a = [star_exposure(p, t, SL, 384, 216) for t in ts]
    b = [star_exposure(p, t, SL, 384, 216) for t in ts[::-1]][::-1]
    assert a == b
    assert all(-2.5 <= v <= 3.5 for v in a)  # key exposure + automatic part within +-3 dex
