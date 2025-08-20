"""
It's supposed to be the equivalent of the "plugin.py" in the NVIDIA variant provider.

The "plugin.py" file for the NVIDIA provider is designed as a highly flexible and extensible framework for resolving complex dependencies, while the AMD provider is a straightforward, single-purpose tool.
`variantlib` is optionally used in the NVIDIA variant provider.

`variantlib` (and its `VariantProperty`) is a tool for managing complex, structured hardware properties, which the sophisticated NVIDIA provider needs, but the straightforward AMD provider does not.
As mentioned, it's a core library that the package manager (like `pip`) would use to interpret the provider's complex output.

TODO:
Flexibility and abstraction similar to NVIDIA's
"""

from __future__ import annotations
import logging
import sys
from typing import Any, List

from .detect_rocm import get_system_info

def get_variants(context: Any) -> List[str]:
    """
    Entry point for the wheel variant provider.

    This function is called by the packaging tool to determine which
    hardware-specific variants are available for the system. It now detects
    both ROCm version and GFX architecture.

    Args:
        context: An object containing information about the request. (Currently unused).

    Returns:
        A list of variant strings, e.g., ['rocm57-gfx90a', 'rocm57'].
    """
    print("AMD ROCm Variant Provider running.", file=sys.stderr)
    system_info = get_system_info()
    
    variants = []
    rocm_version = system_info.get("rocm_version")
    gfx_versions = system_info.get("gfx_versions", [])

    if rocm_version:
        major, minor = rocm_version
        base_rocm_tag = f"rocm{major}{minor}"

        # If we have GFX info, create specific tags for each arch
        # e.g., rocm5.7 on a gfx90a GPU -> "rocm57-gfx90a"
        # A GPU might not report a GFX version (which is possible for non-GPU agents).
        if gfx_versions:
            for gfx in gfx_versions:
                # 
                variants.append(f"{base_rocm_tag}-{gfx}")

        # Always add the base ROCm version as a fallback variant
        variants.append(base_rocm_tag)

    if variants:
        print(f"Identified AMD variants (most specific first): {variants}", file=sys.stderr)
    
    return variants
