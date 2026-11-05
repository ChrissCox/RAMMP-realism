"""Offscreen MuJoCo rendering of clips into MP4 (H.264), GIF and contact-sheet PNG, with overlays:
the clip title, the current beat (mood chip), and a timeline bar coloured by zone with the phase
labels and a playhead. A pair clip shows 'Plain (today)' and 'Puppy Mode' side by side: both start
together, the shorter side holds its last frame, each header runs its own clock.

Also the legibility measure (how far each beat moves the face on screen, in the close camera at
1280x720) and the per-beat peak-frame sheet built on it.
"""

import shutil
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d

from . import scene as scene_mod
from .timeline import DT, FREE, NEAR_USER, PRECISION

MEDIA_DIR = scene_mod.SPIKE_DIR / "media"
WORK_DIR = MEDIA_DIR / "_work"
FAST_DIR = WORK_DIR / "fast"  # --fast previews go here, never over the deliverables
FPS = 25  # every other 50 Hz sample
SAMPLE_STEP = int(round(1.0 / (FPS * DT)))
SINGLE_SIZE = (1280, 720)
PANEL_SIZE = (800, 720)  # each half of a side-by-side
FAST_SCALE = 0.5
CRF = 23
GIF_EVERY = 2  # GIF frames: every other video frame (12.5 fps)
GIF_WIDTH = {"single": 640, "side": 960}
GIF_MAX_MB = 8.0  # a GIF for a document
GIF_COLORS = (256, 192, 128, 96, 64)  # tried in order until the GIF fits GIF_MAX_MB
SHEET_FRAMES = {"single": 12, "side": 8}
SHEET_COLUMNS = {"single": 4, "side": 2}
SHEET_TILE_WIDTH = {"single": 480, "side": 800}
PEAK_COLUMNS = 4
PEAK_TILE_WIDTH = 480
LEGIBILITY_CAMERA = "close"  # legibility is measured here, at SINGLE_SIZE
LEGIBLE_PX = 15.0  # a beat should move the face at least this far on screen
MOUTH_SITES = ("g_left_tip", "g_right_tip")  # the gripper's fingertips: the mouth
FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"

ZONE_COLORS = {FREE: (126, 196, 138), NEAR_USER: (240, 178, 66), PRECISION: (84, 132, 230)}
ZONE_TEXT = {PRECISION: "focus: timing only"}
MOOD_COLORS = {
    "CALM": (148, 163, 184),
    "HELLO": (244, 162, 97),
    "CURIOUS": (250, 196, 62),
    "INTERESTED": (110, 196, 230),
    "FOCUS": (84, 132, 230),
    "PROUD": (167, 125, 230),
    "HAPPY": (240, 120, 150),
    "EAGER": (244, 162, 97),  # the director's moods share their families' colours
    "ATTENTIVE": (110, 196, 230),
    "PUZZLED": (250, 196, 62),
    "SORRY": (150, 140, 200),
    "SHAKE IT OFF": (120, 200, 190),
}
DEFAULT_MOOD_COLOR = (160, 170, 185)
PLATE_RGBA = (14, 18, 26, 178)
TEXT_RGB = (250, 250, 252)
SUBTEXT_RGB = (208, 214, 224)
PAIR_HEADERS = ("Plain (today)", "Puppy Mode")


@dataclass(frozen=True)
class Camera:
    """A free camera (MuJoCo's: azimuth and elevation in degrees). lookat None: the head (site
    'ee') plus head_offset, at the clip's first sample or, with follow_s, following the head of the
    functional motion smoothed over follow_s seconds (so expressions never move the camera)."""

    lookat: tuple | None
    distance: float
    azimuth: float
    elevation: float
    fovy: float
    head_offset: tuple = (0.0, 0.0, 0.0)
    follow_s: float | None = None

    def axes(self):
        """(forward, up) unit vectors of the view, as MuJoCo's free camera has them."""
        az, el = np.radians(self.azimuth), np.radians(self.elevation)
        forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        up = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])
        return forward, up

    def lookats(self, track, kin):
        """(N, 3) where the camera looks at each sample of `track`."""
        if self.lookat is not None:
            return np.tile(np.asarray(self.lookat, dtype=float), (track.n, 1))
        func = track.func if track.func is not None else track
        if self.follow_s is None:
            return np.tile(kin.fk(func.q[0], "ee")[0] + np.asarray(self.head_offset), (track.n, 1))
        head = gaussian_filter1d(kin.fk_track(func.q, "ee")[0], self.follow_s / DT, axis=0, mode="nearest")
        return head + np.asarray(self.head_offset)

    def project(self, points, lookat, size):
        """(N, 2) pixel coordinates of world points (N, 3) in an image of `size` (w, h), the camera
        looking at `lookat` ((3,) or one per point)."""
        forward, up = self.axes()
        rel = np.atleast_2d(points) - (np.atleast_2d(lookat) - self.distance * forward)
        depth = rel @ forward
        focal = size[1] / 2.0 / np.tan(np.radians(self.fovy) / 2.0)
        right = np.cross(forward, up)
        return np.column_stack(
            [size[0] / 2.0 + focal * (rel @ right) / depth, size[1] / 2.0 - focal * (rel @ up) / depth]
        )


CAMERAS = {
    # the whole arm, the user and the counter, 3/4 from the front-right: the face is visible
    "wide": Camera(lookat=(0.36, -0.12, 0.23), distance=1.95, azimuth=146.0, elevation=-17.0, fovy=40.0),
    # the forearm, wrist and face fill the frame, for small expressions (fixed at the first pose)
    "close": Camera(
        lookat=None, distance=0.66, azimuth=148.0, elevation=-5.0, fovy=38.0, head_offset=(-0.06, -0.04, -0.035)
    ),
    # the same framing, following the head (smoothed) so the face stays in frame through moves
    "close_follow": Camera(
        lookat=None,
        distance=0.72,
        azimuth=148.0,
        elevation=-5.0,
        fovy=38.0,
        head_offset=(-0.06, -0.04, -0.035),
        follow_s=0.5,
    ),
    # a reverse angle from beyond the counter: the face while the arm works there, the user behind it
    "counter": Camera(lookat=(0.36, -0.17, 0.34), distance=1.25, azimuth=200.0, elevation=-12.0, fovy=40.0),
    # over the seated user's left shoulder at head height, at the arm's home pose: what the user sees.
    # From home the face looks at the counter, away from the user (about 145 deg off), so here it
    # shows its back and side; a glance at the user turns it only partly this way
    "user_pov": Camera(lookat=(0.12, 0.05, 0.50), distance=0.78, azimuth=24.0, elevation=-8.0, fovy=52.0),
}


# --------------------------------------------------------------------------------- 3-D frames
class Stage:
    """A renderer for one scene, camera and panel size."""

    def __init__(self, scene, camera, size):
        self.scene, self.kin = scene, scene.kin()
        self.width, self.height = size
        self.renderer = mujoco.Renderer(scene.model, self.height, self.width)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = [1, 1, 1, 0, 0, 0]  # hide the clearance proxies and the sites
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.azimuth, self.cam.elevation = camera.distance, camera.azimuth, camera.elevation
        self.fovy = camera.fovy

    def frame(self, q, grip, held, lookat):
        """RGB (H, W, 3) of the scene at one sample, the camera looking at `lookat`."""
        self.kin.pose(q, grip, held)
        self.cam.lookat[:] = lookat
        mujoco.mj_camlight(self.scene.model, self.kin.d)
        self.scene.model.vis.global_.fovy = self.fovy
        self.renderer.update_scene(self.kin.d, self.cam, self.option)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


# ---------------------------------------------------------------------------------- overlays
@lru_cache(maxsize=32)
def _font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def _text_size(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _plate(draw, box, radius, fill=PLATE_RGBA):
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def draw_title(draw, title, subtitle, s):
    """Title and its subtitle lines (a string or a list) top-left on a translucent dark plate (s:
    pixel scale)."""
    lines = [subtitle] if isinstance(subtitle, str) else list(subtitle)
    ft, fs = _font(int(30 * s), True), _font(int(19 * s))
    tw, th = _text_size(draw, title, ft)
    sizes = [_text_size(draw, line, fs) for line in lines]
    line_h = max(h for _, h in sizes) + int(6 * s)
    pad, x, y = int(14 * s), int(22 * s), int(20 * s)
    width = max([tw] + [w for w, _ in sizes])
    _plate(draw, (x, y, x + width + 2 * pad, y + th + len(lines) * line_h + int(2.2 * pad)), int(12 * s))
    draw.text((x + pad, y + pad - int(6 * s)), title, font=ft, fill=TEXT_RGB)
    for k, line in enumerate(lines):
        draw.text((x + pad, y + pad + th + int(4 * s) + k * line_h), line, font=fs, fill=SUBTEXT_RGB)


def draw_mood(draw, mood, right, s):
    """The current beat as a pill at the top right (right: x of its right edge)."""
    if not mood:
        return
    f = _font(int(22 * s), True)
    tw, th = _text_size(draw, mood, f)
    pad_x, pad_y, y = int(18 * s), int(10 * s), int(22 * s)
    color = MOOD_COLORS.get(mood.split(" (")[0], DEFAULT_MOOD_COLOR)
    _plate(draw, (right - tw - 2 * pad_x, y, right, y + th + 2 * pad_y + int(6 * s)), int(22 * s), (*color, 235))
    draw.text((right - tw - pad_x, y + pad_y - int(2 * s)), mood, font=f, fill=(22, 26, 34))


def draw_timeline(draw, track, t, x0, x1, bottom, s, header=None):
    """Zone-coloured bar with phase labels and a playhead; with a header (side by side) the header
    above it carries the side's own clock, which stops at its end."""
    bar_h, label_h = int(20 * s), int(22 * s)
    y1 = bottom - int(18 * s)
    y0 = y1 - bar_h
    band_top = y0 - label_h - int(14 * s) - (int(34 * s) if header else 0)
    _plate(draw, (x0 - int(12 * s), band_top, x1 + int(12 * s), bottom - int(6 * s)), int(10 * s))
    total = max(track.duration, DT)

    def xs(time):
        return x0 + (x1 - x0) * min(max(time / total, 0.0), 1.0)

    _zone_segments(draw, track, xs, y0, y1, s)
    small = _font(int(14 * s))
    t = min(t, track.duration)
    clock = f"{t:4.1f} s / {track.duration:.1f} s"
    label_end = x1 if header else x1 - _text_size(draw, clock, small)[0] - 8 * s  # keep clear of the clock
    last_right = -1e9
    for ph in track.phases:
        a, b = xs(ph.t0), xs(ph.t1)
        draw.line([(a, y0 - int(4 * s)), (a, y1)], fill=(255, 255, 255, 200), width=max(1, int(s)))
        tw, _ = _text_size(draw, ph.name, small)
        if b - a >= tw + 6 * s and a > last_right and a + tw + 6 * s <= label_end:
            draw.text((a + 3 * s, y0 - label_h), ph.name, font=small, fill=TEXT_RGB)
            last_right = a + tw + 6 * s
    px = xs(t)
    draw.line([(px, y0 - int(8 * s)), (px, y1 + int(4 * s))], fill=(255, 255, 255, 255), width=max(2, int(3 * s)))
    r = int(5 * s)
    draw.ellipse((px - r, y0 - int(8 * s) - r, px + r, y0 - int(8 * s) + r), fill=(255, 255, 255, 255))
    if header:
        done = t >= track.duration - 1e-9
        fh = _font(int(20 * s), True)
        draw.text(
            (x0, band_top + int(6 * s)), f"{header}   {t:.1f} s{'  done' if done else ''}", font=fh, fill=TEXT_RGB
        )
        total_text = f"of {track.duration:.1f} s"
        cw, _ = _text_size(draw, total_text, small)
        draw.text((x1 - cw, band_top + int(10 * s)), total_text, font=small, fill=SUBTEXT_RGB)
    else:
        cw, _ = _text_size(draw, clock, small)
        draw.text((x1 - cw, y0 - label_h), clock, font=small, fill=SUBTEXT_RGB)


def _zone_segments(draw, track, xs, y0, y1, s):
    """Runs of equal zone as coloured blocks; PRECISION blocks say 'focus: timing only'."""
    tiny = _font(int(12 * s), True)
    zone = track.zone
    edges = np.flatnonzero(np.diff(zone)) + 1
    starts, ends = np.concatenate([[0], edges]), np.concatenate([edges, [len(zone)]])
    for a, b in zip(starts, ends):
        za, zb = xs(a * DT), xs(b * DT)
        z = int(zone[a])
        draw.rectangle((za, y0, max(za + 1, zb), y1), fill=(*ZONE_COLORS[z], 255))
        text = ZONE_TEXT.get(z)
        if text:
            tw, th = _text_size(draw, text, tiny)
            if zb - za > tw + 8 * s:
                draw.text(((za + zb - tw) / 2, y0 + (y1 - y0 - th) / 2 - 2 * s), text, font=tiny, fill=TEXT_RGB)


def overlay_single(img, title, subtitle, track, t, mood, show_title=True):
    """Title (unless show_title is False), mood chip and timeline on a single-track frame."""
    s = img.width / SINGLE_SIZE[0]
    layer = Image.new("RGBA", img.size)
    draw = ImageDraw.Draw(layer)
    if show_title:
        draw_title(draw, title, subtitle, s)
    draw_mood(draw, mood, img.width - int(22 * s), s)
    draw_timeline(draw, track, t, int(34 * s), img.width - int(34 * s), img.height, s)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def overlay_side(img, title, subtitle, plain, puppy, t, mood, show_title=True):
    """Two panels: a header with its own clock and a timeline per panel, the title (unless
    show_title is False) over the left, the mood on the right."""
    s = img.width / (2 * PANEL_SIZE[0])
    half = img.width // 2
    layer = Image.new("RGBA", img.size)
    draw = ImageDraw.Draw(layer)
    draw.line([(half, 0), (half, img.height)], fill=(255, 255, 255, 230), width=max(2, int(3 * s)))
    if show_title:
        draw_title(draw, title, subtitle, s)
    draw_mood(draw, mood, img.width - int(22 * s), s)
    for k, (track, header) in enumerate(zip((plain, puppy), PAIR_HEADERS)):
        x0 = k * half + int(30 * s)
        draw_timeline(draw, track, t, x0, x0 + half - int(60 * s), img.height, s, header)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


# ------------------------------------------------------------------------------------ frames
class Reel:
    """A clip's tracks on their stages: frame(k) -> (time, overlaid image), any k in any order."""

    def __init__(self, clip, fast=False):
        self.clip = clip
        built = clip.build()
        self.tracks = built if isinstance(built, tuple) else (built,)
        self.subtitle = clip.subtitle_for(self.tracks)
        sc = scene_mod.build(clip.run)
        self.camera = CAMERAS[clip.camera]
        self.kind = "side" if len(self.tracks) == 2 else "single"
        size = PANEL_SIZE if self.kind == "side" else SINGLE_SIZE
        scale = FAST_SCALE if fast else 1.0
        size = tuple(int(v * scale) // 2 * 2 for v in size)
        self.stages = [Stage(sc, self.camera, size) for _ in self.tracks]
        self.lookats = [self.camera.lookats(track, stage.kin) for track, stage in zip(self.tracks, self.stages)]
        n = max(track.n for track in self.tracks)
        self.n_frames = (n + SAMPLE_STEP - 1) // SAMPLE_STEP

    def view(self, i):
        """The 3-D image of output sample i (a shorter track holds its last pose)."""
        views = []
        for track, stage, look in zip(self.tracks, self.stages, self.lookats):
            j = min(i, track.n - 1)
            views.append(stage.frame(track.q[j], track.grip[j], track.held[j], look[j]))
        return Image.fromarray(np.hstack(views))

    def overlaid(self, i, title=True, mood=None):
        """Output sample i with its overlays; mood overrides the chip (default: the beat there)."""
        t = i * DT
        img = self.view(i)
        if mood is None:
            beat = self.tracks[-1].beat_at(t)
            mood = beat.name if beat else self.clip.mood
        if self.kind == "side":
            return overlay_side(img, self.clip.title, self.subtitle, *self.tracks, t, mood, title)
        return overlay_single(img, self.clip.title, self.subtitle, self.tracks[0], t, mood, title)

    def frame(self, k, title=True):
        """(time, image) of video frame k; title=False leaves the title plate off (sheet tiles)."""
        i = k * SAMPLE_STEP
        return i * DT, self.overlaid(i, title)

    def sheet_picks(self):
        """Indices of the evenly spaced frames for the contact sheet."""
        return np.linspace(0, self.n_frames - 1, SHEET_FRAMES[self.kind]).round().astype(int)

    def close(self):
        for stage in self.stages:
            stage.close()


# ------------------------------------------------------------------------------- legibility
@dataclass(frozen=True)
class Legibility:
    """How far one beat moves the face on screen: the peak distance (pixels) between where the
    composed and the functional track put the eye, and the mouth (the farther fingertip), and when."""

    beat: str
    t0: float
    t1: float
    eye_px: float
    mouth_px: float
    peak_t: float


def _face_points(kin, q, grip, held):
    """(N, 1 + len(MOUTH_SITES), 3): the eye and the fingertips along joint rows q, gripper posed."""
    out = np.empty((len(q), 1 + len(MOUTH_SITES), 3))
    for i, (qi, gi, hi) in enumerate(zip(q, grip, held)):
        kin.pose(qi, gi, hi)
        out[i] = [kin.site(s)[0] for s in ("eye", *MOUTH_SITES)]
    return out


def legibility(track, kin, camera=LEGIBILITY_CAMERA, size=SINGLE_SIZE):
    """[Legibility] per beat of a composed track, measured in CAMERAS[camera] at `size`."""
    cam = CAMERAS[camera]
    look = cam.lookats(track, kin)
    func = track.func
    pts, ref = _face_points(kin, track.q, track.grip, track.held), _face_points(kin, func.q, func.grip, func.held)
    moved = np.stack(
        [
            np.linalg.norm(cam.project(pts[:, k], look, size) - cam.project(ref[:, k], look, size), axis=1)
            for k in range(pts.shape[1])
        ],
        axis=1,
    )
    eye, mouth = moved[:, 0], moved[:, 1:].max(axis=1)
    out = []
    for b in track.beats:
        i0, i1 = int(round(b.t0 / DT)), min(int(round(b.t1 / DT)), track.n - 1)
        if i1 <= i0:
            continue
        span = slice(i0, i1 + 1)
        peak = i0 + int(np.argmax(np.maximum(eye[span], mouth[span])))
        out.append(Legibility(b.name, b.t0, b.t1, float(eye[span].max()), float(mouth[span].max()), peak * DT))
    return out


def format_legibility(rows):
    """A fixed-width table of Legibility rows."""
    lines = [f"{'beat':34s} {'t0 s':>6s} {'t1 s':>6s} {'eye px':>7s} {'mouth px':>8s} {'peak s':>7s}"]
    for r in rows:
        flag = "" if max(r.eye_px, r.mouth_px) >= LEGIBLE_PX else f"   < {LEGIBLE_PX:.0f} px"
        lines.append(f"{r.beat:34s} {r.t0:6.1f} {r.t1:6.1f} {r.eye_px:7.1f} {r.mouth_px:8.1f} {r.peak_t:7.2f}{flag}")
    return "\n".join(lines)


def peak_sheet(clip, out_dir=MEDIA_DIR):
    """media/<clip>_peaks.png: one tile per beat at its peak, labelled with the beat and its
    on-screen travel, both measured in the clip's own camera at SINGLE_SIZE (for the close-camera
    reels, the legibility measure itself). Returns (path, [Legibility])."""
    reel = Reel(clip)
    rows = legibility(reel.tracks[-1], reel.stages[-1].kin, clip.camera)
    tiles = []
    for r in rows:
        img = reel.overlaid(int(round(r.peak_t / DT)), title=False, mood="")  # the tile label names it
        tiles.append((f"{r.beat}: eye {r.eye_px:.0f} px, mouth {r.mouth_px:.0f} px at {r.peak_t:.1f} s", img))
    reel.close()
    subtitle = f"{clip.name}: each beat at its peak (on-screen travel in the '{clip.camera}' camera at 1280x720)"
    sheet = out_dir / f"{clip.name}_peaks.png"
    cols, width = (
        (SHEET_COLUMNS["side"], SHEET_TILE_WIDTH["side"]) if reel.kind == "side" else (PEAK_COLUMNS, PEAK_TILE_WIDTH)
    )
    _tile_sheet(clip.title, subtitle, tiles, cols, width).save(sheet)
    return sheet, rows


# ----------------------------------------------------------------------------------- outputs
def render_clip(clip, fast=False):
    """Render a clip to media/<name>.mp4, .gif and _sheet.png; fast: half size, no GIF, into
    FAST_DIR. Returns {kind: path}."""
    out_dir = FAST_DIR if fast else MEDIA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    reel = Reel(clip, fast)
    mp4 = out_dir / f"{clip.name}.mp4"
    gif_dir = WORK_DIR / f"{clip.name}_gif"
    shutil.rmtree(gif_dir, ignore_errors=True)
    gif_dir.mkdir(parents=True)
    picks = set(reel.sheet_picks().tolist())
    tiles, writer = [], None
    for k in range(reel.n_frames):
        t, img = reel.frame(k)
        writer = writer or _mp4_writer(mp4, img.size)
        writer.send(np.asarray(img).tobytes())
        if not fast and k % GIF_EVERY == 0:
            gif_frame(img, reel.kind).save(gif_dir / f"{k // GIF_EVERY:05d}.png", compress_level=1)
        if k in picks:
            tiles.append(reel.frame(k, title=False))
    writer.close()
    reel.close()
    out = {"mp4": mp4, "sheet": contact_sheet(clip, reel.subtitle, reel.kind, tiles, out_dir)}
    if not fast:
        out["gif"] = make_gif(gif_dir, out_dir / f"{clip.name}.gif")
    return out


def sheet_only(clip):
    """Just the contact sheet, rendering only its frames: quick to iterate on a clip's look."""
    reel = Reel(clip)
    tiles = [reel.frame(int(k), title=False) for k in reel.sheet_picks()]
    reel.close()
    return contact_sheet(clip, reel.subtitle, reel.kind, tiles, MEDIA_DIR)


def _mp4_writer(path, size):
    gen = imageio_ffmpeg.write_frames(
        str(path),
        size,
        fps=FPS,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        quality=None,
        macro_block_size=2,
        output_params=["-crf", str(CRF), "-preset", "medium", "-movflags", "+faststart"],
    )
    gen.send(None)
    return gen


def gif_frame(img, kind):
    w = GIF_WIDTH[kind]
    return img.resize((w, round(img.height * w / img.width)), Image.LANCZOS)


def make_gif(frame_dir, out):
    """ffmpeg palettegen/paletteuse over the frames, with fewer colours until it fits GIF_MAX_MB."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    fps = FPS / GIF_EVERY
    for colors in GIF_COLORS:
        graph = (
            f"split[a][b];[a]palettegen=max_colors={colors}:stats_mode=full[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
        )
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            f"{fps}",
            "-i",
            str(frame_dir / "%05d.png"),
            "-filter_complex",
            graph,
            "-loop",
            "0",
            str(out),
        ]
        subprocess.run(cmd, check=True)
        if out.stat().st_size <= GIF_MAX_MB * 1e6:
            break
    return out


def _tile_sheet(title, subtitle, tiles, cols, tile_w):
    """A grid of (label, image) tiles under a title: the PIL image."""
    first = tiles[0][1]
    tile_h = round(first.height * tile_w / first.width)
    rows = (len(tiles) + cols - 1) // cols
    head_h, gap = 64, 8
    sheet = Image.new("RGB", (cols * tile_w + (cols + 1) * gap, head_h + rows * (tile_h + gap) + gap), (24, 28, 36))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap + 4, 8), title, font=_font(28, True), fill=TEXT_RGB)
    draw.text((gap + 4, 40), subtitle, font=_font(16), fill=SUBTEXT_RGB)
    f = _font(15 if tile_w < 600 else 18, True)
    for k, (label, img) in enumerate(tiles):
        x = gap + (k % cols) * (tile_w + gap)
        y = head_h + (k // cols) * (tile_h + gap)
        sheet.paste(img.resize((tile_w, tile_h), Image.LANCZOS), (x, y))
        tw, th = _text_size(draw, label, f)
        draw.rounded_rectangle((x + 8, y + 8, x + tw + 26, y + th + 26), radius=8, fill=(14, 18, 26))
        draw.text((x + 17, y + 12), label, font=f, fill=TEXT_RGB)
    return sheet


def contact_sheet(clip, subtitle, kind, tiles, out_dir):
    """Evenly spaced frames in a grid with their timestamps, under the clip's title."""
    lines = [subtitle] if isinstance(subtitle, str) else subtitle
    labelled = [(f"t = {t:.1f} s", img) for t, img in tiles]
    sheet = _tile_sheet(clip.title, "  |  ".join(lines), labelled, SHEET_COLUMNS[kind], SHEET_TILE_WIDTH[kind])
    path = out_dir / f"{clip.name}_sheet.png"
    sheet.save(path)
    return path


def render_named(name, fast=False):
    """Worker: render the registered clip `name`; (name, {kind: (path, MB)}, seconds)."""
    from .clips import registry

    t0 = time.time()
    out = render_clip(registry()[name], fast)
    return name, {k: (str(path), path.stat().st_size / 1e6) for k, path in out.items()}, time.time() - t0


def sheet_named(name):
    """Worker: the contact sheet of the registered clip `name`; (name, path)."""
    from .clips import registry

    return name, str(sheet_only(registry()[name]))


def peaks_named(name):
    """Worker: the peak-frame sheet of the registered clip `name`; (name, path, [Legibility])."""
    from .clips import registry

    path, rows = peak_sheet(registry()[name])
    return name, str(path), rows
