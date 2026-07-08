# PLY + StyleShot + xatlas — Colab kurulum
import os
import pathlib
import site
from pathlib import Path

!pip install -q "numpy==2.0.2" "scipy==1.14.1"
!pip install -q "diffusers==0.32.2" "transformers==4.46.3" "accelerate==0.34.2" "einops==0.7.0"
!pip install -q opencv-python-headless huggingface_hub safetensors matplotlib tqdm pillow gdown certifi
!pip install -q "open3d>=0.18.0" xatlas trimesh

# Colab SSL duzeltmesi (CERTIFICATE_VERIFY_FAILED)
import os
import ssl

import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context
print("SSL patch OK")

os.chdir("/content")
if not Path("StyleShot").exists():
    !git clone -q https://github.com/open-mmlab/StyleShot.git
os.chdir("/content/StyleShot")

!pip install -q addict future lmdb pyyaml requests yapf
!pip install -q basicsr --no-deps 2>/dev/null || pip install -q git+https://github.com/XPixelGroup/BasicSR.git --no-deps

IP = Path("ip_adapter/ip_adapter.py")
t = IP.read_text(encoding="utf-8")
t = t.replace(
    "mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1",
    "mult = len(controlnet.nets) if hasattr(controlnet, 'nets') else 1",
)
t = t.replace(
    "if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):",
    "if hasattr(controlnet, 'nets') and isinstance(controlnet_conditioning_scale, float):",
)
t = t.replace(
    "controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)",
    "controlnet_keep.append(keeps[0] if not hasattr(controlnet, 'nets') else keeps)",
)
t = t.replace(
    "if isinstance(self.pipe.controlnet, MultiControlNetModel):",
    "if hasattr(self.pipe.controlnet, 'nets'):",
)
OLD_GP = """        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )"""
NEW_GP = """        if hasattr(controlnet, 'nets'):
            global_pool_conditions = controlnet.nets[0].config.global_pool_conditions
        else:
            global_pool_conditions = getattr(controlnet.config, 'global_pool_conditions', False)"""
t = t.replace(OLD_GP, NEW_GP)
t = t.replace(
    "        if isinstance(controlnet, ControlNetModel):\n            image = self.prepare_image(",
    "        if not hasattr(controlnet, 'nets'):\n            image = self.prepare_image(",
)
t = t.replace(
    "        elif isinstance(controlnet, MultiControlNetModel):\n            images = []",
    "        else:\n            images = []",
)
t = t.replace("        else:\n            assert False\n", "")
IP.write_text(t, encoding="utf-8")

for sp in site.getsitepackages():
    bsr = pathlib.Path(sp) / "basicsr"
    if not bsr.exists():
        continue
    for f in bsr.rglob("*.py"):
        txt = f.read_text(encoding="utf-8", errors="ignore")
        new = txt.replace(
            "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
            "from torchvision.transforms.functional import rgb_to_grayscale",
        )
        if new != txt:
            f.write_text(new, encoding="utf-8")

import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())

import sys
sys.path.insert(0, "/content/g3d")
try:
    from vrcaps_colab_bootstrap import bootstrap
    orbit, _ = bootstrap()
    print("Script kaynagi (GitHub):", orbit)
except ImportError:
    print("GitHub bootstrap yok — once Bolum 0 calistirin")

