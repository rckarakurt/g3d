# 0 — GitHub'dan VRCaps reposunu klonla / guncelle
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ============ AYARLAR ============
# Public repo (push ettikten sonra URL'yi guncelleyin):
REPO_URL = "https://github.com/rckarakurt/g3d.git"
REPO_BRANCH = "main"

# Private repo — Colab: Secrets -> GITHUB_TOKEN, sonra:
# from google.colab import userdata
# REPO_URL = f"https://{userdata.get('GITHUB_TOKEN')}@github.com/rckarakurt/g3d.git"

REPO_DIR = Path("/content/g3d")
# ================================


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


if REPO_DIR.exists() and (REPO_DIR / ".git").is_dir():
    _run(["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", REPO_BRANCH])
    _run(["git", "-C", str(REPO_DIR), "checkout", REPO_BRANCH])
    _run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", REPO_BRANCH])
    print("Repo guncellendi:", REPO_DIR)
else:
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    _run(["git", "clone", "-b", REPO_BRANCH, "--depth", "1", REPO_URL, str(REPO_DIR)])
    print("Repo klonlandi:", REPO_DIR)

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
