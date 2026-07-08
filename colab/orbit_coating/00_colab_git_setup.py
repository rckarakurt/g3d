# 0 — GitHub'dan VRCaps reposunu klonla / guncelle
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# ============ AYARLAR ============
REPO_URL = "https://github.com/rckarakurt/g3d.git"
REPO_BRANCH = "main"
REPO_DIR = Path("/content/g3d")

# Sifirdan baslarken True yap — eski view_bank / ciktilari siler
CLEAN_OLD_OUTPUTS = True
# ================================


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    return result


def _clone() -> None:
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    _run(
        [
            "git",
            "clone",
            "-b",
            REPO_BRANCH,
            "--depth",
            "1",
            REPO_URL,
            str(REPO_DIR),
        ]
    )
    print("Repo klonlandi:", REPO_DIR)


def _update() -> None:
    """Shallow clone icin pull yerine fetch + reset (--ff-only Colab'da sik kirilir)."""
    try:
        _run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_BRANCH, "--depth", "1"])
        _run(["git", "-C", str(REPO_DIR), "checkout", "-f", REPO_BRANCH])
        _run(["git", "-C", str(REPO_DIR), "reset", "--hard", f"origin/{REPO_BRANCH}"])
        print("Repo guncellendi:", REPO_DIR)
    except subprocess.CalledProcessError:
        print("fetch/reset basarisiz — yeniden klonlaniyor...")
        _clone()


if REPO_DIR.exists() and (REPO_DIR / ".git").is_dir():
    _update()
else:
    _clone()

os.environ["VRCAPS_REPO"] = str(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))

from vrcaps_colab_bootstrap import bootstrap

orbit, gaze = bootstrap()
if not (orbit / "coating_utils.py").exists():
    raise FileNotFoundError(
        f"Scriptler bulunamadi: {orbit}\n"
        "REPO_URL ve REPO_BRANCH dogru mu? Once GitHub'a push edin."
    )

from colab_content_paths import print_repo_info

if CLEAN_OLD_OUTPUTS:
    import shutil as _shutil

    for _p in (
        Path("/content/ply_styleshot_out"),
        Path("/content/gaze_composite"),
        Path("/content/medical_gan_dataset"),
    ):
        if _p.exists():
            _shutil.rmtree(_p)
            print("Silindi:", _p)

_commit = subprocess.run(
    ["git", "-C", str(REPO_DIR), "log", "-1", "--oneline"],
    capture_output=True,
    text=True,
    check=False,
)
if _commit.stdout.strip():
    print("Git commit:", _commit.stdout.strip())

from turntable_render import wall_azimuth_grid, wall_orbit_camera

print("Aci konvansiyonu: Y ekseni, -90..+90, 0=duz yuz (+Z)")
print("Ornek grid:", wall_azimuth_grid(45, 90))
for _az in (-90, 0, 90):
    _c2w = wall_orbit_camera(_az, obliquity_deg=18.0, distance=1.0, target=np.zeros(3))
    _eye = _c2w[:3, 3]
    print(f"  az {_az:+4d} -> eye ({_eye[0]:+.3f}, {_eye[1]:+.3f}, {_eye[2]:+.3f})")

print_repo_info()
print("\nSonraki adim: 1 -> 1b -> 2 -> 5b -> 3 -> 4 -> 6 -> 7")
