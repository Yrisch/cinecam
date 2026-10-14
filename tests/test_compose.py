# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""Gas colours, proxy splats, cores and labels (render/compose.py)."""

import numpy as np
import pytest

from cinecam.path.camera import CamState, pixel_size
from cinecam.render.compose import compose, core_radius, gas_rgb, lut, splat_column

LOOK = {"cmap": "inferno", "show_gas": True, "show_cores": True, "core_radius": None, "core_color": [1.0, 0.27, 0.0]}


def view():
    return CamState(pos=[0, -10, 0], target=[0, 0, 0], up=[0, 0, 1], clim=(0.0, 2.0))


def test_gas_rgb_maps_the_log_range_to_the_colormap():
    rgb = gas_rgb(np.array([[1e-5, 1.0, 100.0, 1e9]]), (0.0, 2.0), "inferno")
    np.testing.assert_array_equal(rgb[0, 0], lut("inferno")[0])
    np.testing.assert_array_equal(rgb[0, 1], lut("inferno")[0])
    np.testing.assert_array_equal(rgb[0, 2], lut("inferno")[255])
    np.testing.assert_array_equal(rgb[0, 3], lut("inferno")[255])


def test_splat_conserves_mass():
    """Particles in the plane of the target: sum of column x pixel area = total mass."""
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(-1, 1, 5000), np.zeros(5000), rng.uniform(-1, 1, 5000)]
    st = view()
    img = splat_column(pts, 0.01, st, 200, 100, smooth=0.7)
    area = pixel_size(st, 10.0, 100) ** 2
    assert img.sum() * area == pytest.approx(50.0, rel=1e-6)


def test_core_radius_physical_or_override():
    sinks = {"core_racc": np.array([0.02, 0.03])}
    np.testing.assert_array_equal(core_radius(LOOK, sinks), [0.02, 0.03])
    assert core_radius(dict(LOOK, core_radius=0.3), sinks) == 0.3
    np.testing.assert_array_equal(core_radius(dict(LOOK, core_radius=0.0), sinks), [0.02, 0.03])


def test_compose_draws_cores_stars_and_label():
    st = view()
    sigma = np.full((100, 200), 10.0)
    sinks = {"core_pos": np.array([[0.0, 0.0, 0.0]]), "core_racc": np.array([0.5])}
    out = compose(sigma, st, LOOK, sinks, None, None)
    assert out.shape == (100, 200, 3) and out.dtype == np.uint8
    np.testing.assert_array_equal(out[50, 100], [255, 69, 0])  # core colour at the centre
    assert (out[5, 5] == np.round(lut("inferno")[127] * 255)).all()  # gas half way in the log range
    stars = np.zeros((100, 200, 3))
    stars[10, 10] = 1.0
    assert (compose(sigma, st, LOOK, sinks, stars, None)[10, 10] == 255).all()  # screen blend
    dots = np.zeros((100, 200, 4))
    dots[10, 10] = [0.5, 0.25, 0.0, 0.5]  # premultiplied RGBA: half opaque (1, 0.5, 0)
    gas = lut("inferno")[127].astype(float)
    expected = np.round((gas * 0.5 + [0.5, 0.25, 0.0]) * 255 + 1e-9)
    np.testing.assert_allclose(compose(sigma, st, LOOK, sinks, dots, None)[10, 10], expected, atol=1)  # drawn over
    lab = compose(sigma, st, LOOK, None, None, "t = 1.00 Myr")
    changed = (lab != compose(sigma, st, LOOK, None, None, None)).any(axis=2)
    assert changed[60:, 100:].any() and not changed[:60].any()  # text bottom right
    assert lab[changed].max() > 240  # in white
    dark = compose(sigma, st, dict(LOOK, show_gas=False, show_cores=False), sinks, None, None)
    assert dark.max() == 0
