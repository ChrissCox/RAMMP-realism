"""Offscreen MuJoCo rendering of clips into MP4 (H.264), GIF and contact-sheet PNG, with overlays:
the clip title, the current beat (mood chip), and a timeline bar coloured by zone with the phase
labels and a playhead. Side-by-side clips show 'Plain (today)' and 'Puppy Mode' together.
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
GIF_MAX_MB = 12.0
GIF_COLORS = (256, 192, 128, 96)  # tried in order until the GIF fits GIF_MAX_MB
SHEET_FRAMES = {"single": 12, "side": 8}
SHEET_COLUMNS = {"single": 4, "side": 2}
SHEET_TILE_WIDTH = {"single": 480, "side": 800}
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


@dataclass(frozen=True)
class Camera:
    """A free camera. lookat None means: the head at the clip's first sample, plus head_offset."""

    lookat: tuple | None
    distance: float
    azimuth: float
    elevation: float
    fovy: float
    head_offset: tuple = (0.0, 0.0, 0.0)


CAMERAS = {
    # the whole arm, the user and the counter, 3/4 from the front-right: the face is visible
    "wide": Camera(lookat=(0.36, -0.12, 0.23), distance=1.95, azimuth=146.0, elevation=-17.0, fovy=40.0),
    # the forearm, wrist and face fill the frame, for small expressions
    "close": Camera(
        lookat=None, distance=0.66, azimuth=148.0, elevation=-5.0, fovy=38.0, head_offset=(-0.06, -0.04, -0.035)
    ),
}


# --------------------------------------------------------------------------------- 3-D frames
class Stage:
    """A renderer for one scene, camera and panel size."""

    def __init__(self, scene, camera, size, first_q):
        self.scene, self.kin = scene, scene.kin()
        self.width, self.height = size
        self.renderer = mujoco.Renderer(scene.model, self.height, self.width)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = [1, 1, 1, 0, 0, 0]  # hide the clearance proxies and the sites
        self.cam = mujoco.MjvCamera()
        lookat = camera.lookat
        if lookat is None:
            lookat = self.kin.fk(first_q, "ee")[0] + np.array(camera.head_offset)
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = camera.distance, camera.azimuth, camera.elevation
        self.fovy = camera.fovy

    def frame(self, q, grip, held):
        """RGB (H, W, 3) of the scene at one sample."""
        self.kin.pose(q, grip, held)
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
    """Title and one-line subtitle top-left on a translucent dark plate (s: pixel scale)."""
    ft, fs = _font(int(30 * s), True), _font(int(19 * s))
    tw, th = _text_size(draw, title, ft)
    sw, sh = _text_size(draw, subtitle, fs)
    pad, x, y = int(14 * s), int(22 * s), int(20 * s)
    _plate(draw, (x, y, x + max(tw, sw) + 2 * pad, y + th + sh + int(2.4 * pad)), int(12 * s))
    draw.text((x + pad, y + pad - int(6 * s)), title, font=ft, fill=TEXT_RGB)
    draw.text((x + pad, y + pad + th + int(4 * s)), subtitle, font=fs, fill=SUBTEXT_RGB)


def draw_mood(draw, mood, right, s):
    """The current beat as a pill at the top right (right: x of its right edge)."""
    if not mood:
        return
    f = _font(int(22 * s), True)
    tw, th = _text_size(draw, mood, f)
    pad_x, pad_y, y = int(18 * s), int(10 * s), int(22 * s)
    color = MOOD_COLORS.get(mood, DEFAULT_MOOD_COLOR)
    _plate(draw, (right - tw - 2 * pad_x, y, right, y + th + 2 * pad_y + int(6 * s)), int(22 * s), (*color, 235))
    draw.text((right - tw - pad_x, y + pad_y - int(2 * s)), mood, font=f, fill=(22, 26, 34))


def draw_timeline(draw, track, t, x0, x1, bottom, s, header=None):
    """Zone-coloured bar with phase labels and a playhead; optional header above it (side-by-side)."""
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
    last_right = -1e9
    for ph in track.phases:
        a, b = xs(ph.t0), xs(ph.t1)
        draw.line([(a, y0 - int(4 * s)), (a, y1)], fill=(255, 255, 255, 200), width=max(1, int(s)))
        tw, _ = _text_size(draw, ph.name, small)
        if b - a >= tw + 6 * s and a > last_right:
            draw.text((a + 3 * s, y0 - label_h), ph.name, font=small, fill=TEXT_RGB)
            last_right = a + tw + 6 * s
    px = xs(t)
    draw.line([(px, y0 - int(8 * s)), (px, y1 + int(4 * s))], fill=(255, 255, 255, 255), width=max(2, int(3 * s)))
    r = int(5 * s)
    draw.ellipse((px - r, y0 - int(8 * s) - r, px + r, y0 - int(8 * s) + r), fill=(255, 255, 255, 255))
    clock = f"{t:4.1f} s / {track.duration:.1f} s"
    cw, _ = _text_size(draw, clock, small)
    if header:
        fh = _font(int(20 * s), True)
        draw.text((x0, band_top + int(6 * s)), header, font=fh, fill=TEXT_RGB)
        draw.text((x1 - cw, band_top + int(10 * s)), clock, font=small, fill=SUBTEXT_RGB)
    else:
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


def overlay_single(img, clip, track, t, mood, title=True):
    """Title (unless title=False), mood chip and timeline on a single-track frame."""
    s = img.width / SINGLE_SIZE[0]
    layer = Image.new("RGBA", img.size)
    draw = ImageDraw.Draw(layer)
    if title:
        draw_title(draw, clip.title, clip.subtitle, s)
    draw_mood(draw, mood, img.width - int(22 * s), s)
    draw_timeline(draw, track, t, int(34 * s), img.width - int(34 * s), img.height, s)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def overlay_side(img, clip, plain, puppy, t, mood, title=True):
    """Two panels: headers and timelines per panel, the title (unless title=False) over the left,
    the mood on the right."""
    s = img.width / (2 * PANEL_SIZE[0])
    half = img.width // 2
    layer = Image.new("RGBA", img.size)
    draw = ImageDraw.Draw(layer)
    draw.line([(half, 0), (half, img.height)], fill=(255, 255, 255, 230), width=max(2, int(3 * s)))
    if title:
        draw_title(draw, clip.title, clip.subtitle, s)
    draw_mood(draw, mood, img.width - int(22 * s), s)
    for k, (track, header) in enumerate(((plain, "Plain (today)"), (puppy, "Puppy Mode"))):
        x0 = k * half + int(30 * s)
        draw_timeline(draw, track, min(t, track.duration), x0, x0 + half - int(60 * s), img.height, s, header)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


# ------------------------------------------------------------------------------------ frames
class Reel:
    """A clip's tracks on their stages: frame(k) -> (time, overlaid image), any k in any order."""

    def __init__(self, clip, fast=False):
        self.clip = clip
        built = clip.build()
        sc = scene_mod.build(clip.run)
        camera = CAMERAS[clip.camera]
        scale = FAST_SCALE if fast else 1.0
        self.tracks = built if isinstance(built, tuple) else (built,)
        self.kind = "side" if len(self.tracks) == 2 else "single"
        size = PANEL_SIZE if self.kind == "side" else SINGLE_SIZE
        size = tuple(int(v * scale) // 2 * 2 for v in size)
        self.stages = [Stage(sc, camera, size, track.q[0]) for track in self.tracks]
        n = max(track.n for track in self.tracks)
        self.n_frames = (n + SAMPLE_STEP - 1) // SAMPLE_STEP

    def frame(self, k, title=True):
        """(time, image) of video frame k; title=False leaves the title plate off (sheet tiles)."""
        i = k * SAMPLE_STEP
        t = i * DT
        views = []
        for track, stage in zip(self.tracks, self.stages):
            j = min(i, track.n - 1)  # a shorter track holds its last pose
            views.append(stage.frame(track.q[j], track.grip[j], track.held[j]))
        img = Image.fromarray(np.hstack(views))
        mood_track = self.tracks[-1]
        beat = mood_track.beat_at(t)
        mood = beat.name if beat else self.clip.mood
        if self.kind == "side":
            return t, overlay_side(img, self.clip, *self.tracks, t, mood, title)
        return t, overlay_single(img, self.clip, self.tracks[0], t, mood, title)

    def sheet_picks(self):
        """Indices of the evenly spaced frames for the contact sheet."""
        return np.linspace(0, self.n_frames - 1, SHEET_FRAMES[self.kind]).round().astype(int)

    def close(self):
        for stage in self.stages:
            stage.close()


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
            _gif_frame(img, reel.kind).save(gif_dir / f"{k // GIF_EVERY:05d}.png", compress_level=1)
        if k in picks:
            tiles.append(reel.frame(k, title=False))
    writer.close()
    reel.close()
    out = {"mp4": mp4, "sheet": contact_sheet(clip, reel.kind, tiles, out_dir)}
    if not fast:
        out["gif"] = make_gif(gif_dir, out_dir / f"{clip.name}.gif")
    return out


def sheet_only(clip):
    """Just the contact sheet, rendering only its frames: quick to iterate on a clip's look."""
    reel = Reel(clip)
    tiles = [reel.frame(int(k), title=False) for k in reel.sheet_picks()]
    reel.close()
    return contact_sheet(clip, reel.kind, tiles, MEDIA_DIR)


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


def _gif_frame(img, kind):
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


def contact_sheet(clip, kind, tiles, out_dir):
    """Evenly spaced frames in a grid with their timestamps, under the clip's title."""
    cols, tile_w = SHEET_COLUMNS[kind], SHEET_TILE_WIDTH[kind]
    first = tiles[0][1]
    tile_h = round(first.height * tile_w / first.width)
    rows = (len(tiles) + cols - 1) // cols
    head_h, gap = 64, 8
    sheet = Image.new("RGB", (cols * tile_w + (cols + 1) * gap, head_h + rows * (tile_h + gap) + gap), (24, 28, 36))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap + 4, 8), clip.title, font=_font(28, True), fill=TEXT_RGB)
    draw.text((gap + 4, 40), clip.subtitle, font=_font(16), fill=SUBTEXT_RGB)
    for k, (t, img) in enumerate(tiles):
        x = gap + (k % cols) * (tile_w + gap)
        y = head_h + (k // cols) * (tile_h + gap)
        sheet.paste(img.resize((tile_w, tile_h), Image.LANCZOS), (x, y))
        label = f"t = {t:.1f} s"
        f = _font(18, True)
        tw, th = _text_size(draw, label, f)
        draw.rounded_rectangle((x + 8, y + 8, x + tw + 26, y + th + 26), radius=8, fill=(14, 18, 26))
        draw.text((x + 17, y + 12), label, font=f, fill=TEXT_RGB)
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
