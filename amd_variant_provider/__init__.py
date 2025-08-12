"""
AMD ROCm Variant Provider for WheelNext
"""

from .plugin import get_variants

# This is the primary entry point that packaging tools (e.g., `pip`) will look for.
# It acts as a standard Python package entry point.
# It points to the function that provides the variant tags.
#
# In contrast, "nvidia_variant_provider/__init__.py" adopts a defensive technique called "vendoring".
# It is to manipulate Python's import system. It finds its private, bundled copy of the packaging library
# (located in "nvidia_variant_provider/vendor/packaging/") and loads it into memory.
# This way guarantees that the provider uses a specific version of the packaging library,
# so that it's compatible, preventing conflicts with any other versions that might be installed on a system.
# As the implementation of NVIDIA WheelNext variant provider relies on external Python dependencies,
# it may have to vendor the packaging library.
# Requiring external Python dependencies may create a chicken-and-egg problem:
# the variant provider runs before pip installs packages,
# so it cannot rely on pip to install its own dependencies.
AMD_VARIANT_PROVIDER_ENTRY_POINT = "amd_variant_provider.plugin:get_variants"

__all__ = ["get_variants", "AMD_VARIANT_PROVIDER_ENTRY_POINT"]
__version__ = "0.0.1"
