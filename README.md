# cinecam — camera paths through Phantom simulations, rendered by shamrock

```bash
./cinecam ../esc_00*                        # editor, dumps in time order (globs ok)
./cinecam ../esc_00* --path orbit.json      # open / create a path file
./cinecam render orbit.json --scale 0.5     # batch render (draft at half res)
./cinecam render orbit.json --frames 0:150 --sycl 0:0 --mp4 movie.mp4
./cinecam render orbit.json --stereo        # flat-screen stereo pair (even files left eye, odd right)
```

Working on the code (or letting an AI agent do it)? Read [`CLAUDE.md`](CLAUDE.md) first:
architecture, invariants of the path interpolation, how to extend and test.

## Installation

cinecam is pure Python; nothing to compile besides shamrock itself.

1. **Build shamrock with its python module** (see the shamrock documentation). The build
   directory must contain `pysham/` and the `libsham*.so` libraries. The launcher looks
   for it in `../../build` (cinecam sitting in `<shamrock>/tmp/cinecam`); otherwise set
   `SHAMROCK_BUILD`.
2. **Create a python environment** with the dependencies. Use the same python version
   as the one shamrock was built against, or `import shamrock` will fail:

   ```bash
   python3 -m venv ~/venvs/cinecam
   ~/venvs/cinecam/bin/pip install numpy scipy matplotlib pillow PyQt6 pytest
   ```

   `ffmpeg` on the `PATH` is only needed for `render --mp4`.
3. **Tell the launcher which python to use.** It runs `$CINECAM_PYTHON`, or `python3`
   from the `PATH` when it is not set (so an activated venv works):

   ```bash
   # once, in ~/.bashrc (or ~/.zshrc)
   export CINECAM_PYTHON=~/venvs/cinecam/bin/python
   export SHAMROCK_BUILD=/path/to/shamrock/build   # only if not ../../build
   ```

4. **Check**:

   ```bash
   B=${SHAMROCK_BUILD:-../../build}
   PYTHONPATH=$B/pysham LD_LIBRARY_PATH=$B ${CINECAM_PYTHON:-python3} \
       -c "import shamrock, PyQt6, scipy, matplotlib, PIL; print('ok')"
   ```

   then `./cinecam your_dumps_*`. Optionally link the launcher into your `PATH`
   (`ln -s $PWD/cinecam ~/.local/bin/cinecam`); it resolves its own location.

Caches: sink trajectories go to `~/.cache/cinecam/` (`$XDG_CACHE_HOME`), editor settings
to the Qt settings of the user.

## Editor

| area | what |
|---|---|
| left, **world view** | simulation overview (particle splat, stars, cores), the path (coloured by time), every keyframe as a frustum with its number, the live camera in white. Drag the dots to move camera positions, the crosses to move targets (Ctrl: both). |
| right, **viewfinder** | what the camera sees. Instant particle proxy while you move, then the real shamrock render (gas + stars + cores, same compositor as the movie) ~0.2 s after you stop. |
| **timeline** | keyframes (drag to retime, Shift+drag to stretch the segment before a key, right-click to delete), simulation dump track, playhead (click / drag to scrub). Stretched segments are drawn in orange with their factor. |
| **camera inspector** | exact numbers for the camera at the playhead + per-key look (Σ range, star exposure, extinction). |
| **movie & look** | duration, fps, output resolution, preview resolution, colormap, layers, star PSF on/off (off: the proxy dots), spikes, bright star glow, auto exposure, time label size. |

Started without dumps (`./cinecam`), the editor asks for them: select the dumps, or a
camera path `.json`, which is opened with the dumps it lists.

Workflow: scrub to a time, fly the camera in the viewfinder, press **K**.
When the playhead is on a keyframe, all edits change that keyframe.
`Edit → Insert orbit…` generates a turn (or a spiral with *end distance* < 1, or a
time-lapse with *end dump index*) starting from the current camera.

**Stretching a segment** (`Edit → Segment duration at playhead…`, Ctrl+T, right-click on
the timeline between two keys, or Shift+drag a key): changes the time between two keys
without touching the rest of the movie. The keys after it move with it, everything before
and after plays exactly as before (only later), and the camera enters and leaves the
segment at the same speed; inside it, the camera takes the extra time in the middle, and
the simulation dumps go by slower. Past about 3× the camera would have to turn back to
keep its speed, so it slows down at the ends of the segment instead. Shortening works
the same way.

Viewfinder: LMB orbit · Shift+LMB/MMB pan · RMB/Ctrl+LMB look around · wheel dolly ·
Ctrl+wheel fov · Alt+wheel dolly-zoom · WASD/RF fly · Q/E roll (Shift ×5).
AZERTY keyboards: `View → Keyboard layout` or `--keyboard azerty` (remembered) gives
ZQSD/RF fly · A/E roll, and &/"/è for the 1/3/7 world views.
World view: click an end of the axis gizmo (bottom left) to look from that side (+X, -Y, ...),
click it again for the opposite side; 1/3/7 front/side/top, F frame all.
Global: K key · Del delete key · Ctrl+T segment duration · Space play · Shift+Space play until the next key · ←/→ frame · PgUp/PgDn keys · Esc reset the
live camera to the path · Ctrl+Z/Ctrl+Shift+Z undo/redo · Ctrl+E export full-res frame.

## Instant views and the proxy cache

The viewfinder proxy and the world view splat a sample of 400k gas particles per dump
(`--sample`). The samples are read straight from the dump files by a fast reader (only
the positions, ~0.1-0.2 s per dump, no shamrock) in a background thread, kept in memory
(`--proxy-cache`, MB, default 1024: ~210 dumps) and prefetched: the current dump first,
then the dumps the movie visits, nearest in movie time first (ahead of the playhead
first). A green strip under the timeline dump track marks the frames whose gas is in
memory. Scrubbing and playing (~15 fps) show the gas of every frame at once; shamrock
only loads a dump for the real preview when you stop, and for the renders.

The sample is the 400k living particles with the lowest fixed random rank of their id
(`iorig` in Phantom): the same particles from one dump to the next, an accreted one is
replaced by the next ranked one, so the proxy does not flicker.

### Dump formats

The fast readers (gas sample for the proxy, sinks for the trajectories) are pluggable:
`src/cinecam/data/readers.py` defines `DumpReader` (`read_gas`, `read_sinks`) and the `READERS`
registry. `phantom` is the only one so far and the default; the format is stored in the
path file (`"format"`) and can be set with `--format`. Supporting another code (gadget,
shamrock dumps, ...) means adding a reader there. The real renders still load the dumps
through shamrock (`src/cinecam/render/backend.py`).

## Following a star or a core

Click a star / core in the viewfinder or the world view to select it (white ring), then
in the camera panel:

* **Look at**: the camera stays where it is and keeps the sink at the centre (tripod).
* **Ride along**: the camera moves with the sink, keeping its offset (chase camera); you can
  still orbit / dolly around it.
* **Stop**: back to a world camera.

Press K to key it. Keys that follow a sink have a cyan ring on the timeline, the followed
sink a cyan ring in the views (editor only, never rendered). Each frame (the world and
every followed sink) has its own smooth curves through the keys, and the camera blends
from one key's frame to the next one's, so following, switching from one sink to another
(the target glides to the new sink while the camera moves to its new offset) and
releasing are all smooth. Changing the dump of a following key moves it with its sink.

Sink trajectories are read from the sink block of every dump in the background (a few
seconds for 1000 dumps, cached in `~/.cache/cinecam/`). A sink is its index in the dump
sink arrays, which Phantom keeps across dumps. Between two dumps the sinks follow a cubic
Hermite curve built from their positions and velocities; stars and cores are also drawn
there in the renders, so a followed sink stays exactly where the camera expects it.

## Path file (JSON)

```json
{ "fps": 30, "duration": 12, "resolution": [1920, 1080],
  "dumps": ["/abs/esc_00400", "/abs/esc_00500"], "format": "phantom",
  "look": {"cmap": "inferno", "show_stars": true, "show_cores": true, "core_radius": null, ...},
  "keyframes": [{"t": 0, "pos": [..], "target": [..], "up": [..], "fov": 45, "dump": 0,
                 "clim": [0.1, 3.5], "exposure": 0, "extinction": 1,
                 "follow": null, "ride": false, "ease": false, "stretch": 1}, ...] }
```

Interpolation is built for shooting keys one after the other: the path is a chain of
segments, one cubic per pair of consecutive keys, joined with the same position and the
same velocity (C1). The velocity at a key is estimated from that key and the ones before
it, so

* **adding a key at the end never changes the segments already shot**;
* editing a key only changes the segments around it (the one before it and the three after it:
  the velocity at a key comes from the two keys before it).

Two cases look at the next key and so also reshape the end of the previous segment, because
the camera must come to rest there: a *hold* (consecutive keys with the same position, or
target, or up: the camera stays exactly still between them, e.g. same place on different
dumps), and the distance / fov / Σ range / exposure turning back (they never overshoot).

Each key has an **ease** checkbox (camera panel): zero velocity at that key, the camera
settles into it and starts again from rest (first and last keys of a shot, a pause).

A key's `stretch` is the factor its segment (to the next key) was stretched by: the
velocities are estimated as if the segment had kept its original duration, which is why
a stretched segment changes nothing around it.

The target and the direction of `pos - target` are interpolated this way, the camera
distance through its log (orbits stay circular; *Insert orbit* puts a key every 15° for an
even angular speed), `dump` is linear and rounded to the nearest dump, and a change of
followed sink is a smooth ramp between two keys.

The `up` vector is stored as typed (e.g. exactly `0 0 1`); shamrock's Camera3d makes it
orthogonal to the view direction. The `+X/+Y/+Z` buttons next to it (and *Level horizon*,
= +Z) set it exactly. Orbit drags and head turns pivot around this up vector, so a +Z
camera stays level; only a roll (Q/E) changes it.

## Rendering

* gas: `model.render_column_integ("rho")` with `shamrock.math.Camera3d_f64_3`, log Σ in M☉/pc².
* stars (sinks with accretion radius < 0.01 pc), in true colour:
  * luminosity, radius, temperature from the mass (main sequence relations); black body
    through the CIE 1931 colour matching functions → linear sRGB (a lookup table in T);
  * flux at the true distance from the camera, dimmed by the gas column between camera and
    star (two shamrock half-ray integrals, Cardelli extinction at the sRGB wavelengths,
    `extinction` multiplier per key);
  * analytic PSF stamped at each star's sub-pixel position: Moffat core (FWHM 1.6 px on a
    1920 px frame, ×`psf_scale`, wider in red), Moffat halo, optional diffraction spikes
    (`star_spikes`, turned by `spike_angle` degrees counterclockwise on screen); each stamp
    only as large as the visible part of that star's PSF; sizes scale with the frame width, so a draft and the final movie look the same;
  * colour preserving asinh stretch of the intensity (Lupton et al. 2004), highlights fade
    to white; `exposure` (dex, per key) shifts it;
  * `glow_gamma` (*bright star glow γ* in the Movie panel, default 1): tames the stars
    brighter than white. With the physical flux (γ = 1) a star seen 100 times closer is
    10⁴ times brighter: its centre is white anyway, so only its halo and spikes grow, up
    to a large white disc. With γ < 1 a star whose peak pixel would be *x* times white
    is drawn *x*^γ times white instead (colour kept): its glow still grows on approach,
    but slowly (0.5: 100 times over white, 0.3: 16 times, for a star 10⁴ times over).
    Stars below white are unchanged, and the limit follows the exposure, so it works
    with auto exposure;
  * `star_psf` off: stars as round dots, exactly as in the instant proxy view (same brightness,
    colour, size and opacity, drawn over the gas; sizes relative to the frame width, so the
    proxy, the preview and the movie match);
  * `auto_exposure`: the exposure follows the brightest stars in view (see
    [Star exposure](#star-exposure)); the key exposure is added to it.
* cores (other sinks): orange discs of their physical size, the accretion radius stored in the
  dump (4000 au here). `core_radius` (code length units, 0 / `null` = physical) overrides it.
* time label (`time_label`): simulation time in the bottom right corner, `label_size` px
  high on a 1080 px tall frame (default 36), scaled with the output height.

### Star exposure

The star brightness on screen is set by an exposure in dex (a factor 10 per unit): the
star fluxes are multiplied by 10^exposure before the asinh stretch, which is linear for
faint pixels, logarithmic for bright ones and white at `I_WHITE` (`render/stars.py`).

* **Manual** (default): the `exposure` of the keys, interpolated along the path like the
  other per-key values.
* **Auto** (`auto_exposure`): an automatic exposure is added to the key one, so the key
  exposure becomes a correction (+1: ten times brighter than automatic). At time *t*:
  1. for 11 times spread over [*t* − 0.5 s, *t* + 0.5 s] of the path (clipped to the movie),
     take the camera of that time and the stars of its dump;
  2. keep the stars whose centre is in the frame, in front of the camera;
  3. compute the peak pixel intensity of each one (flux at its distance × the PSF core
     peak, grey average of R, G, B);
  4. choose the exposure that puts the **10th brightest** at half the white intensity,
     just below white after the stretch (with fewer than 10 stars in view, the faintest
     one). So a few very bright stars saturate and the bulk of the field stays visible;
  5. average those exposures with Gaussian weights (σ = a quarter of the window), skipping
     times with no star in view, and clamp to ±3 dex (0 when there is no star at all).

  The result depends on the path time only, so frames can be rendered in any order or in
  parallel and give the same exposure (and both stereo eyes share it). The averaging
  smooths the changes over about a second, like an eye adapting, not instantly.

  Limits of this first version: the gas extinction is ignored when choosing the exposure
  (a star hidden in a dense core still counts as bright), `psf_scale` is not taken into
  account, stars just outside the frame do not count even if their glow does, and when
  the stars leave the view the automatic part jumps back to 0 (smoothed over the window).

### Stereo (flat screen)

`./cinecam render path.json --stereo [angle]`, or the *stereo pair* box of the render
dialog, renders each frame of the path twice: path frame *i* becomes file `2i` (left eye)
and `2i+1` (right eye), at the same size as a mono frame. `--frames` still counts path
frames; `--skip-existing` skips the eyes already on disk; `--mp4` is not available in
stereo. Nothing changes in the path file: any mono path can be rendered in stereo.

The two eyes are derived from the path's camera at render time:

* seen from the camera target they are `angle` degrees apart (default 4°, ±2° per eye),
  side by side along the camera's right vector, so the eye separation scales with the
  distance to the target and the depth effect stays the same through a dolly;
* they look **parallel** to the mono camera (no toe-in, which would add vertical
  offsets between the eyes towards the corners), and each image is shifted sideways
  (an off-axis view, rendered a few % wider and cropped) so that the **target is at the
  centre of both images**: it sits on the screen plane, what is behind it appears behind
  the screen, what is in front comes out;
* the star exposure (auto exposure included) is the mono camera's, the same for both eyes,
  and the time label is at the same place in both.

On a large screen keep the angle moderate: gas far behind the target is up to about
`angle / horizontal fov` of the image width apart between the eyes. VR stereo
(fixed eye separation, headset projection) is a different mode, not done yet.

## Layout

```
cinecam                 launcher (editor, `render`, `test`)
examples/orbit.json     a 360° orbit path (open it with your dumps: ./cinecam dumps... --path examples/orbit.json)
src/cinecam/
  constants.py          units, star / core threshold
  path/                 camera.py (camera maths, projection), campath.py (keys, interpolation, following, files)
  data/                 readers.py (fast dump readers), sinktracks.py (sink trajectories), proxycache.py (proxy samples)
  render/               compose.py (gas, cores, labels), stars.py (star layer), backend.py (shamrock scene, render process)
  gui/                  app.py (main window), worldview, viewfinder, timeline, inspectors, workers, keymaps, qtutil
  render_movie.py       batch renderer (./cinecam render)
tests/                  pytest suite, with a writer of small synthetic Phantom dumps
```

## Tests

```bash
./cinecam test                                   # everything that needs neither shamrock nor a dump
CINECAM_TEST_DUMP=../esc_00500 ./cinecam test    # + the shamrock tests on a real dump
./cinecam test -m "not gui"                      # without the (offscreen) editor tests
```

The tests run on small synthetic Phantom dumps (`tests/phantom.py`): camera maths, the serial
interpolation (adding / editing a key only changes the path next to it, holds, ease,
no overshoot), following sinks, the dump reader, sink tracks, the proxy sample and cache,
the star photometry / PSF / stretch / auto exposure, compositing, and the editor
(offscreen, with a stub of the render process). `pytest` is the only extra dependency.

## License

cinecam is licensed under the CeCILL Free Software License Agreement v2.1, like shamrock:
see [LICENSE](LICENSE) (French version) or [LICENSE.en](LICENSE.en) (English version).

SPDX-License-Identifier: CECILL-2.1
