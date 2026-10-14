# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""Keyboard layouts of the fly / roll / view keys (QWERTY, AZERTY)."""

from PyQt6.QtCore import Qt

# fly / roll keys at the same physical positions on both layouts
KEYMAPS = {
    "qwerty": {
        "fly": {"w": (1, 0, 0), "s": (-1, 0, 0), "a": (0, -1, 0), "d": (0, 1, 0), "r": (0, 0, 1), "f": (0, 0, -1)},
        "roll": {"q": 1, "e": -1},
        "views": {},
    },
    "azerty": {
        "fly": {"z": (1, 0, 0), "s": (-1, 0, 0), "q": (0, -1, 0), "d": (0, 1, 0), "r": (0, 0, 1), "f": (0, 0, -1)},
        "roll": {"a": 1, "e": -1},
        "views": {"&": "1", '"': "3", "è": "7"},  # unshifted digit row
    },
}


def keymap_hint(name):
    return "ZQSD/RF fly · A/E roll" if name == "azerty" else "WASD/RF fly · Q/E roll"


def key_char(ev):
    """Layout independent letter of a key event (Qt.Key follows the active keyboard layout)."""
    k = ev.key()
    if Qt.Key.Key_A.value <= k <= Qt.Key.Key_Z.value:
        return chr(k).lower()
    return ev.text().lower()
