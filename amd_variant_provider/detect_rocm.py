"""
This file is dedicated to the logic of detecting the installed ROCm environment (the equivalent of `class CudaEnvironment` defined in the file "detect_cuda.py" in the repo "nvidia-variant-provider").
So, the system driver (i.e., AMDGPU equivalent to the NVIDIA KMD), the ROCm UMD driver, and the GFX arch (AMD's equivalent of NVIDIA GPU Compute Capability) need to be detected.
Then, wheels that are specifically compiled for different ROCm environments can be automatically picked.
Examples of GFX Name: `gfx9xx` for Instinct MI-series, `gfx1xxx` for Navi GPUs.

Notes of AMD KMD:
1) AMDGPU; 2) AMDKFD (a component of `amdgpu`); 3) AMD unified drivers.

Similar approaches are done in PyTorch "torch/utils/cpp_extension.py"

At this time, we are not using such things as `pyrsmi` (that is the AMD equivalent of NVIDIA `pynvml`) to avoid external Python dependencies.
We thought the initial implementation of WheelNext variant provider should be lightweight, dependency-free, and non-intrusive.
Based on the "Right Tool for the Job" principle, using `pyrsmi` for now for this task is like using a sledgehammer to hang a picture frame.

The current method relies on the `rocminfo` command which is standard in any ROCm install.
"""

from __future__ import annotations
import logging
import os
import platform
import re
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

# Draft that is expected to be the equivalent of `class CudaEnvironment`.
# Unused yet.
@dataclass(frozen=True)
class ROCmEnvironment:
    kernel_module_version: Optional[str]  # `modinfo amdgpu | grep -Ei '^version:'
    #rocm_version: Optional[str]  # `rocminfo`
    rocm_version: Optional[tuple[int, int]]  # `rocminfo`
    gfx_archs: List[str]  # `rocminfo` or `rocm_agent_enumerator -name`

def _get_amdgpu_kmd_version() -> Optional[str]:
    """
    Detects the version of the installed AMDGPU KMD.

    It first attempts to read the version directly from the pseudo file system "sysfs" (`/sys/`),
    which is the most efficient method. If that fails, it falls back to
    parsing the output of the `modinfo` command.

    Returns:
        The version string of the KMD (e.g., "6.7.99") or None if not found.
    """
    # Strategy 1: Read directly from the `/sys/` (most efficient w/o launching a subprocess)
    try:
        version_path = Path("/sys/module/amdgpu/version")
        if version_path.is_file():
            kmd_version = version_path.read_text(encoding="utf-8").strip()
            # Basic validation to ensure the file isn't empty
            if kmd_version:
                return kmd_version
    except (IOError, OSError):
        # This can happen if there are permission issues or the file doesn't exist.
        pass  # Silently fall through to the next strategy.

    # Strategy 2: Fallback to parsing `modinfo` (more robust)
    try:
        result = subprocess.run(
            ["modinfo", "amdgpu"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        match = re.search(r"^version:\s*(.*)", result.stdout, re.MULTILINE)
        if match:
            return match.group(1).strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # This can happen if `modinfo` isn't in the PATH or the module doesn't exist.
        return None

    return None

# Regex for ROCm version (like "5.7" or "6.4.3", only major/minor)
_ROCM_VERSION_REGEX = re.compile(r"ROCm(?:\s+Version)?[:\s]*(\d+)\.(\d+)(?:\.\d+)?", re.IGNORECASE)
# RegEx for GFX inspired by https://github.com/ROCm/rocminfo/blob/c34ac33d661bd2c87d9c3b956eb8b15ac8f7092c/rocm_agent_enumerator#L95
#_GFX_REGEX = re.compile(r"^\s*Name:\s+(gfx\d+[0-9a-f]*)", re.MULTILINE)
_GFX_REGEX = re.compile(r"(gfx[0-9a-fA-F]+(?:-[0-9a-fA-F]+)?(?:-generic)?(?:[:][-+:\w]+)?)")

def _get_gfx_from_agent_enumerator() -> list[str]:
    exec_path = shutil.which("rocm_agent_enumerator")
    if not exe:
        return []
    try:
        # The `-name` option prints just the architecture names; it's designed for scripts.
        # Debian manpage: rocm_agent_enumerator(1)
        # ROCm docs: "prints list of available architecture names".
        proc = subprocess.run(
            [exec_path, "-name"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        vals = []
        for line in proc.stdout.splitlines():
            tok = _normalize_gfx(line)
            m = _GFX_REGEX.search(line)
            tok = None if not m else m.group(1).lower()
            # "gfx000" represents CPU. Keep GPUs only.
            if tok and tok != "gfx000":
                vals.append(tok)
        return sorted(set(vals))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []

def _get_info_from_rocminfo() -> Dict[str, Any]:
    """
    Runs `rocminfo` once and parses it for both ROCm version (UMD) and GFX versions.
    Other commands (e.g., `rocm-smi`) or lib APIs can be alternatives for detection.
    Possible more detection strategies:
    - Package manager queries (dpkg, rpm, pacman)
    - ROCm library presence checking (librocm, libhip)
    - Environment variable inspection (HIP_PLATFORM, ROCM_VERSION)
    - CMake cache examination
    - ROCm SMI tool usage

    Returns:
        A dictionary containing "rocm_version" (e.g., (5, 7)) and 
        "gfx_names" (e.g., ["gfx90a", "gfx1030"]).
    """
    if platform.system() != "Linux":
        logger.info("ROCm detection skipped: not running on Linux")
        return {}

    info = {}

    exec_path = shutil.which("rocminfo")
    if not exec_path:
        return info

    try:
        # The `rocminfo` tool is one of the standard ways to get system-level ROCm details.
        output = subprocess.check_output(
            [exec_path],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=7,  # No need for large timeouts because of confirmed existence
        )

        # Find ROCm version
        version_match = _ROCM_VERSION_REGEX.search(output)
        if version_match:
            major, minor = map(int, version_match.groups())
            info["rocm_version"] = (major, minor)
            logging.info(f"Found rocm{major}.{minor} via `rocminfo`.")

        # Find all unique GFX versions.
        # FIXME:
        # `rocm_agent_enumerator -name` may be better than `rocminfo` because it prints clear `gfx*` names.
        gfx_matches = _GFX_REGEX.findall(output)
        if gfx_matches:
            # FIXME:
            # Brittle and NOT future-proof.
            # MI and Navi
            # https://github.com/pytorch/pytorch/blob/4d5f92aa39d294a833038299aa3f38f99ebc31b6/.ci/docker/manywheel/build.sh#L86
            # Also refer to https://d2awnip2yjpvqn.cloudfront.net/v2 (internal only?)
            GFX_CODE = os.environ.get("AMD_PREFERRED_GFX_ARCHS", "").split(',') or
                ["gfx900", "gfx906", "gfx908", "gfx90a", "gfx942", "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201"]
            # Use a set to store unique GFX versions, then sort for deterministic output.
            # TODO: prioritized list may be needed.
            unique_gfx = sorted(set(gfx_matches))
            if all(x in GFX_CODE for x in unique_gfx):
                info["gfx_names"] = unique_gfx
                logging.info(f"Found GFX architectures: {unique_gfx} via `rocminfo`.")

    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e:
        logging.error(f"Could not run or parse `rocminfo`: {e}")

    return info

def _get_rocm_version_from_dir(rocm_path_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    (Fallback) Attempts to infer the ROCm version from the version file in a given directory.
    """
    rocm_path: Optional[Path] = None
    if rocm_path_str:
        rocm_path = Path(rocm_path_str)
    else:
        path_str_from_env = os.environ.get("ROCM_PATH")
        if path_str_from_env:
            rocm_path = Path(path_str_from_env)
    if not rocm_path:
        rocm_path = Path("/opt/rocm")

    # A happy path for checking ROCm version on Linux
    # Refs.: https://rocmdocs.amd.com/projects/rccl/en/latest/how-to/troubleshooting-rccl.html
    version_file = rocm_path / ".info" / "version"
    if version_file.is_file():
        try:
            content = version_file.read_text(encoding="utf-8").strip()
            # The version file typically just contains "x.y.z"
            parts = content.split('.')
            if len(parts) >= 2:
                major, minor = int(parts[0]), int(parts[1])
                logging.info(f"Found rocm{major}.{minor} via version file: {version_file}")
                return major, minor
        except (ValueError, IOError) as e:
            logging.error(f"Error reading ROCm version file: {e}")
    return None

# TODO (optional): `apt show rocm-libs -a` as a distro-specific fallback.

@lru_cache(maxsize=1)
def get_system_info() -> Dict[str, Any]:
    """
    Detects installed ROCm version and GFX architectures using multiple strategies.

    Strategies, in order of preference:
    1. Run the `rocminfo` command (Linux only).
    2. Check the `ROCM_PATH` environment variable for a version file.
    3. Check the default installation path "/opt/rocm" for a version file.

    Returns:
        A dictionary with "rocm_version" and "gfx_names".
    """
    # Strategy 1: Use `rocminfo` to get everything at once
    info = _get_info_from_rocminfo()

    # Fallback for ROCm version if `rocminfo` failed or didn't find it
    if "rocm_version" not in info:
        # Strategy 2: Check `ROCM_PATH` environment variable
        rocm_path_env = os.environ.get("ROCM_PATH")
        version = _get_rocm_version_from_dir(rocm_path_env)
        if version:
            info["rocm_version"] = version
    if "gfx_names" not in info:
        from_agent = _get_gfx_from_agent_enumerator()
        if from_agent:
            info["gfx_names"] = from_agent

    if not info:
        logging.warning("Neither ROCm version nor GFX architecture could be detected.")
    else:
        # This "kmd_version" may be not needed for the current simple, straightforward AMD WheelNext variant provider.
        info["kmd_version"] = _get_amdgpu_kmd_version()

    return info
