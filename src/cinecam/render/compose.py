# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Frame compositing for cinecam: gas column density colormap, proxy splats, cores, labels.

Only numpy / scipy / matplotlib (colormaps) / PIL: used both by the render server
(shamrock process) and by the editor (proxy previews).
"""

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from ..path.camera import ZNEAR, pixel_size, project

_LUTS = {}


def lut(name):
    if name not in _LUTS:
        import matplotlib

        _LUTS[name] = (matplotlib.colormaps[name](np.linspace(0, 1, 256))[:, :3]).astype(np.float32)
    return _LUTS[name]


def gas_rgb(sigma, clim, cmap):
    """Column density [Msun/pc^2] -> float RGB in [0,1], log scale between 10**clim."""
    lo, hi = clim
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (np.log10(np.maximum(sigma, 1e-40)) - lo) / max(hi - lo, 1e-6)
    idx = (np.clip(np.nan_to_num(v, nan=0.0), 0, 1) * 255).astype(np.int32)
    return lut(cmap)[idx]


def screen_blend(a, b):
    return 1.0 - (1.0 - a) * (1.0 - b)


# ---------------------------------------------------------------------- proxy


def splat_column(pts, weights, state, nx, ny, smooth=0.7):
    """
    Cheap column density estimate from a particle subsample: each particle deposits
    weight / (pixel area at its depth) in its pixel. Same units as the real render
    if weights are particle masses in Msun and positions in code units with the
    area converted by the caller.
    """
    sx, sy, z = project(state, pts, nx, ny)
    ok = (z > ZNEAR) & (sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny)
    ix = sx[ok].astype(np.int64)
    iy = sy[ok].astype(np.int64)
    area = pixel_size(state, z[ok], ny) ** 2
    w = weights[ok] if np.ndim(weights) else np.full(ix.shape, float(weights))
    img = np.bincount(iy * nx + ix, weights=w / area, minlength=nx * ny).reshape(ny, nx)
    if smooth:
        img = gaussian_filter(img, smooth)
    return img

# ---------------------------------------------------------------------- cores


def draw_discs(img, sx, sy, rad, color, alpha=1.0):
    """Anti-aliased filled discs on a float RGB image, in place."""
    ny, nx, _ = img.shape
    color = np.asarray(color, dtype=float)
    for x, y, r in zip(sx, sy, rad):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        r = max(r, 0.8)
        x0, x1 = int(max(0, math.floor(x - r - 1))), int(min(nx, math.ceil(x + r + 1)))
        y0, y1 = int(max(0, math.floor(y - r - 1))), int(min(ny, math.ceil(y + r + 1)))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.hypot(xx + 0.5 - x, yy + 0.5 - y)
        cov = np.clip(r + 0.5 - dist, 0.0, 1.0)[..., None] * alpha
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - cov) + color * cov


def core_discs(state, pos, nx, ny, radius):
    """radius: scalar or one value per core (code units)."""
    sx, sy, z = project(state, pos, nx, ny)
    ok = z > ZNEAR
    rad = np.broadcast_to(radius, z.shape) / pixel_size(state, np.where(ok, z, 1.0), ny)
    return sx[ok], sy[ok], rad[ok]


def core_radius(look, sinks):
    """World radius of the core discs: look["core_radius"] if set, else the sink accretion radii."""
    r = look.get("core_radius")
    if r is None or r <= 0:
        return sinks.get("core_racc", 0.04)
    return r


# ---------------------------------------------------------------------- full frame


def compose(sigma, state, look, sinks=None, star_layer=None, label=None):
    """
    Final RGB uint8 frame.

    sigma: (ny, nx) column density [Msun/pc^2], top row first.
    sinks: dict with "core_pos" (and "core_racc") in code units if cores are drawn.
    star_layer: float RGB from StarLight.render or None.
    """
    ny, nx = sigma.shape
    img = gas_rgb(sigma, state.clim, look["cmap"]) if look.get("show_gas", True) else np.zeros((ny, nx, 3), np.float32)
    img = img.astype(np.float64)
    if look.get("show_cores", True) and sinks is not None and len(sinks["core_pos"]):
        sx, sy, rad = core_discs(state, sinks["core_pos"], nx, ny, core_radius(look, sinks))
        draw_discs(img, sx, sy, rad, look.get("core_color", (1.0, 0.27, 0.0)))
    if star_layer is not None:
        img = screen_blend(img, star_layer)
    out = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    if label:
        out = draw_label(out, label)
    return out


def draw_label(img, text):
    from PIL import Image, ImageDraw, ImageFont

    ny, nx, _ = img.shape
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    size = max(4, ny // 30)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    w = dr.textlength(text, font=font)
    dr.text((nx - w - size, ny - 2 * size), text, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def save_png(img, filename):
    from PIL import Image

    Image.fromarray(img).save(filename)
