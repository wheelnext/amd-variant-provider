"""
WheelNext Variant Provider for AMD ROCm"
"""

# This is the primary entry point that packaging tools (e.g., `pip`) will look for.
# It acts as a standard Python package entry point.
# It points to the class that provides the variant tags.
#
# In contrast, "nvidia_variant_provider/__init__.py" in the GitHub repo "nvidia-variant-provider" adopts a defensive technique called "vendoring".
# It is to manipulate/hijack Python's import system. It finds its private, bundled copy of the packaging library
# (located in "nvidia_variant_provider/vendor/packaging/") and loads it into memory.
# This way guarantees that the provider uses a specific version of the packaging library,
# so that it's compatible, preventing conflicts with any other versions that might be installed on a system.
#
# As the implementation of NVIDIA WheelNext variant provider relies on external Python dependencies,
# it may have to vendor the packaging library.
#
# Requiring external Python dependencies may create a chicken-and-egg problem:
# the variant provider runs before pip installs packages (the so-called "pre-pip" execution),
# so it cannot rely on `pip` to install its own dependencies.
#
# In the initial (but production-ready) implementation of AMD WheelNext Variant Provider plugin,
# as there are no external Python dependencies, no need to do vendoring.
__version__ = "0.0.1"
__description__ = "A WheelNext Variant Provider for AMD ROCm"
__all__ = ["__version__", "__description__"]
