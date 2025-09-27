"""
It's supposed to be the equivalent of the "plugin.py" in the NVIDIA variant provider, which detects AMD hardware and driver characteristics and then generates a prioritized list of "features" that a package manager like `pip` can use to select the best possible wheel.

The "plugin.py" file for the NVIDIA provider is designed as a highly flexible and extensible framework for resolving complex dependencies, while the AMD provider is a straightforward, single-purpose tool.

Notes:
1. During the evolution of WheelNext variant provider, there were two types of providers - simple, string-based provider vs. complex, config-/class-based provider.
2. As of 08/27/2025, the latest spec (https://wheelnext.dev/proposals/pepxxx_wheel_variant_support/) only has the complex, config-/class-based provider left.
3. AMD WheelNext variant provider was initially implemented following the approach to the (deprecated) simple, string-based provider.
4. As the spec evolved further, it then switched to the approach of the complex, config-/class-based provider.

TODO:
Flexibility and abstraction similar to NVIDIA's
"""

from __future__ import annotations
import logging
import os
import sys
import re
from dataclasses import dataclass
from functools import cached_property
from typing import Any, List, Protocol, runtime_checkable

from amd_variant_provider.detect_rocm import get_system_info, ROCmEnvironment, AMDVariantFeatureKey, ROCmVersion

logger = logging.getLogger(__name__)

# `variantlib` (and its `VariantProperty`) is a tool for managing complex, structured hardware properties.
# It's a core library that the package manager (like `pip`) would use to interpret the provider's complex output.
# However, `variantlib` is optionally used. Why it's optional is to avoid a direct dependency on it, which makes the provider self-contained.
# Such a dependency is risky for a low-level packaging plugin. It could create version conflicts or complicate the bootstrap process where `pip` needs to install and run the provider before other packages.
# `class VariantPropertyType(Protocol)` is thus defined and acts as a local blueprint or contract, to describe the shape of the data expected.
@runtime_checkable
class VariantPropertyType(Protocol):
    @property
    def namespace(self) -> str:
        """Namespace (from plugin)"""
        raise NotImplementedError

    @property
    def feature(self) -> str:
        """Feature name (within the namespace)"""
        raise NotImplementedError

    @property
    def value(self) -> str:
        """Feature value"""
        raise NotImplementedError

# This `dataclass` defines the structure that `pip` expects.
# It must have a `.name` and a `.values` attribute.
@dataclass(frozen=True)
class VariantFeatureConfig:
    name: str
    # Acceptable values in priority order for this feature
    values: list[str]
    multi_value: bool = False

class AMDVariantPlugin:
    """
    The AMD ROCm Variant Provider Plugin.
    This class implements the interface expected by the WheelNext standard.
    """
    namespace = "amd"
    is_build_plugin = False

    # WheelNext static plugin API
    dynamic = False

    @staticmethod
    def _parse_list_env(env_val: str | None) -> list[str]:
        if not env_val:
            return []
        # Split on comma/space/semicolon, trim, drop empties.
        parts = re.split(r"[,\s;]+", env_val.strip())
        return [p for p in (s.strip() for s in parts) if p]

    @cached_property
    def _system_info(self) -> dict[str, Any]:
        """
        Probe the system once and cache the result.
        """
        return get_system_info()

    def get_supported_configs(self) -> list[VariantFeatureConfig]:
        """
        This is the standardized method that `pip` will call.
        """
        logger.info(f"[{self.namespace}-variant-provider] Running system detection.")

        configs: list[VariantFeatureConfig] = []

        # TODO: Prioritized list of GFX archs.
        # E.g., dGPU might be preferred over iGPU; PCIe vs. APU.
        env_gfx = os.environ.get("AMD_VARIANT_PROVIDER_FORCE_GFX_ARCH")
        if not env_gfx:
            gfx_archs = self._system_info.get(AMDVariantFeatureKey.GFX_ARCH)
        else:
            gfx_archs = self._parse_list_env(env_gfx);

        # Priority 1: GFX architecture (most specific)
        if gfx_archs:
            # The list of all detected GFX architectures is provided.
            # The dependency resolver will try to match them.
            #configs.append(
            #    VariantFeatureConfig(name=AMDVariantFeatureKey.GFX_ARCH, values=gfx_archs)
            #)
            configs.append(
                VariantFeatureConfig(name=AMDVariantFeatureKey.GFX_ARCH, values=gfx_archs, multi_value=True)
            )
        # Priority 2: ROCm version (more general)
        # Env var is type `str`
        if rocm_version_env := os.environ.get("AMD_VARIANT_PROVIDER_FORCE_ROCM_VERSION", None):
            rocm_version_list = rocm_version_env.strip().split('.')
            assert(len(rocm_version_list) == 3)
            rocm_version = ROCmVersion(*rocm_version_list)
        else:
            rocm_version = self._system_info.get(AMDVariantFeatureKey.ROCM_VERSION)
        if rocm_version:
            configs.append(
                VariantFeatureConfig(name=AMDVariantFeatureKey.ROCM_VERSION, values=[f"{rocm_version.major}.{rocm_version.minor}"], multi_value=False)
            )

        if configs:
            logger.info(f"[{self.namespace}-variant-provider] Detected features: {configs}")
        else:
            logger.warning(f"[{self.namespace}-variant-provider] No AMD features detected.")

        return configs

    def get_all_configs(self) -> list[VariantFeatureConfig]:
        return [
                VariantFeatureConfig(name=AMDVariantFeatureKey.ROCM_VERSION, values=["6.4", "6.3"], multi_value=False),
                VariantFeatureConfig(name=AMDVariantFeatureKey.GFX_ARCH, values=["gfx900", "gfx906", "gfx908", "gfx90a", "gfx942", "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201"], multi_value=True),
        ]


def main() -> int:
    """Minimal CLI to print detected configs for debugging)."""
    logging.basicConfig(level=os.environ.get("AMD_VARIANT_PROVIDER_LOGLEVEL", "INFO"))
    plugin = AMDVariantPlugin()

    def print_all_configs() -> None:
      cfgs = plugin.get_all_configs()
      for c in cfgs:
          print(f"{plugin.namespace} :: {c.name} :: {c.values}")

    print_all_configs()

    def print_supported_configs() -> None:
      cfgs = plugin.get_supported_configs()
      for c in cfgs:
          print(f"{plugin.namespace} :: {c.name} :: {c.values}")

    print_supported_configs()

    os.environ["AMD_VARIANT_PROVIDER_FORCE_GFX_ARCH"] = "gfx1100"
    print_supported_configs()

    os.environ["AMD_VARIANT_PROVIDER_FORCE_ROCM_VERSION"] = "7.0.0"
    print_supported_configs()

    return 0

if __name__ == "__main__":
    sys.exit(main())
