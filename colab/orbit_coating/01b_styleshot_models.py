# 1b — StyleShot HF modelleri (ip.bin + SD1.5 + CLIP, ~8 GB)
import os
import sys
from pathlib import Path

sys.path.insert(0, "/content/g3d")
from vrcaps_colab_bootstrap import bootstrap

bootstrap()
from colab_content_paths import install_sys_path

install_sys_path()

os.chdir("/content/StyleShot")

from styleshot_models import ensure_styleshot_models, ip_bin_path

ensure_styleshot_models()
print("ip.bin:", ip_bin_path())

