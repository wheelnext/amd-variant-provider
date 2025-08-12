"""
This file is dedicated to the logic of detecting the installed ROCm version.
From the perspectives of logics and semantics, it's the equivalent of "detect_cuda.py".
In addition, we need to detect the GFX arch as well, for wheels that are specifically compiled for different GPU microarchitectures, such as Instinct MI-series (`gfx9xx`) and consumer Navi (`gfx1xxx`) GPUs.

Similar approaches used in PyTorch "torch/utils/cpp_extension.py"

At this time, we are not using such things as `pyrsmi` (that is the equivalent of NVIDIA `pynvml`) to avoid external Python dependencies.
We thought the initial implementation of WheelNext variant provider should be lightweight, dependency-free, and non-intrusive.
Based on the "Right Tool for the Job" principle, using `pyrsmi` for now for this task is like using a sledgehammer to hang a picture frame.

The current method relies on the `rocminfo` command which is standard in any ROCm install.
"""

import os
import re
import sys
import platform
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

def _log(msg: str):
    """Logs a message to stderr for debugging purposes."""
    print(f"[{__name__}] {msg}", file=sys.stderr)

def _get_info_from_rocminfo() -> Dict[str, Any]:
    """
    Runs `rocminfo` once and parses it for both ROCm version and GFX versions.
    Other commands (e.g., `rocm-smi`) or lib APIs can be alternatives for detection.

    Returns:
        A dictionary containing "rocm_version" (e.g., (5, 7)) and 
        "gfx_versions" (e.g., ["gfx90a", "gfx1030"]).
    """
    if platform.system() != "Linux":
        return {}

    # Regex for ROCm version and GFX name
    ROCM_VERSION_REGEX = re.compile(r"ROCm Version:\s*(\d+)\.(\d+)")
    GFX_NAME_REGEX = re.compile(r"^\s*Name:\s+(gfx\d+[0-9a-f]*)", re.MULTILINE)

    info = {}
    try:
        # The `rocminfo` tool is one of the standard ways to get system-level ROCm details.
        output = subprocess.check_output(
            ["rocminfo"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10, # Increased timeout slightly for potentially long output
        )

        # Find ROCm version
        version_match = ROCM_VERSION_REGEX.search(output)
        if version_match:
            major, minor = map(int, version_match.groups())
            info['rocm_version'] = (major, minor)
            _log(f"Found ROCm {major}.{minor} via rocminfo.")

        # Find all unique GFX versions
        gfx_matches = GFX_NAME_REGEX.findall(output)
        if gfx_matches:
            # Use a set to store unique GFX versions, then sort for deterministic output
            unique_gfx = sorted(list(set(gfx_matches)))
            info['gfx_versions'] = unique_gfx
            _log(f"Found GFX architectures: {unique_gfx} via rocminfo.")

    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        _log(f"Could not run or parse `rocminfo`: {e}")

    return info

def _get_rocm_version_from_dir(rocm_path_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    (Fallback) Attempts to infer the ROCm version from the version file in a given directory.
    """
    if not rocm_path_str:
        return None

    rocm_path = Path(rocm_path_str)
    version_file = rocm_path / ".info" / "version"
    if version_file.is_file():
        try:
            content = version_file.read_text(encoding="utf-8").strip()
            # The version file typically just contains "x.y.z"
            parts = content.split('.')
            if len(parts) >= 2:
                major, minor = int(parts[0]), int(parts[1])
                _log(f"Found ROCm {major}.{minor} via version file: {version_file}")
                return major, minor
        except (ValueError, IOError) as e:
            _log(f"Error reading ROCm version file: {e}")
    return None

def get_system_info() -> Dict[str, Any]:
    """
    Detects installed ROCm version and GFX architectures using multiple strategies.

    Strategies, in order of preference:
    1. Run the `rocminfo` command (Linux only).
    2. Check the `ROCM_PATH` environment variable for a version file.
    3. Check the default installation path "/opt/rocm" for a version file.

    Returns:
        A dictionary with 'rocm_version' and 'gfx_versions'.
    """
    # Strategy 1: Use `rocminfo` to get everything at once
    info = _get_info_from_rocminfo()

    # Fallback for ROCm version if `rocminfo` failed or didn't find it
    if 'rocm_version' not in info:
        # Strategy 2: Check `ROCM_PATH` environment variable
        rocm_path_env = os.environ.get("ROCM_PATH")
        version = _get_rocm_version_from_dir(rocm_path_env)
        if version:
            info['rocm_version'] = version
        else:
            # Strategy 3: Check default installation path /opt/rocm (common on Linux)
            if platform.system() == "Linux":
                version = _get_rocm_version_from_dir("/opt/rocm")
                if version:
                    info['rocm_version'] = version

    if not info:
        _log("Neither ROCm version nor GFX architecture could be detected.")

    return info
