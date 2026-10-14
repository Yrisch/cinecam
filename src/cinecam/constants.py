# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""Physical constants and conventions shared by the modules (cgs unless stated)."""

MSUN_G = 1.9891e33  # solar mass [g]
PC_CM = 3.086e18  # parsec [cm]
MYR_S = 3.15576e13  # megayear [s]
AU_PER_PC = 206264.8

STAR_MAX_RACC_PC = 0.01  # sinks with a smaller accretion radius [pc] are stars, the others cores
