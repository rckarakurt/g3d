# 0 — GitHub'dan VRCaps reposunu klonla / guncelle
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ============ AYARLAR ============
REPO_URL = "https://github.com/rckarakurt/g3d.git"
REPO_BRANCH = "main"
REPO_DIR = Path("/content/g3d")
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

print_repo_info()
print("\nSonraki adim: Bolum 1 (StyleShot kurulum)")
