"""Smoke test of the INSTALLED puppy_motion. CI runs it as `cd /tmp && python3 -I <this file>` so
nothing from the repo checkout can shadow the package."""

import importlib.metadata as md
import sys
from importlib import resources

import puppy_motion
from puppy_motion.flatbot import chair_scene

where = puppy_motion.__file__
if "site-packages" not in where and "dist-packages" not in where:
    sys.exit(f"puppy_motion is not the installed package: {where}")
dist = md.distribution("puppy-motion")  # raises if `pip show puppy-motion` would fail
if dist.metadata["Name"].upper() == "UNKNOWN" or dist.version == "0.0.0":
    sys.exit(f"empty UNKNOWN distribution: {dist.metadata['Name']} {dist.version}")
scene = chair_scene()
if not scene.person or not scene.obstacles:
    sys.exit("chair_scene() is empty: package data missing")
if not resources.files("puppy_motion").joinpath("data/zones.yaml").read_text(encoding="utf-8"):
    sys.exit("data/zones.yaml is empty")
print(f"installed puppy_motion {dist.version} ok: {where}")
