"""Minimal helpers for StyleShot Colab notebooks."""

import numpy as np
import matplotlib.pyplot as plt


def disable_nsfw_filter(pipe):
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    return pipe
