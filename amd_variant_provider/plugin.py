"""
It's supposed to be the equivalent of the "plugin.py" in the NVIDIA variant provider, which detects AMD hardware and driver characteristics and then generates a prioritized list of "features" that a package manager like `pip` can use to select the best posssible wheel.

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

from .detect_rocm import get_system_info, ROCmEnvironment

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

# Standardized feature keys
class AMDVariantFeatureKey:
    # DRIVER = "kmd_version"
    ROCm = "rocm_version"
    GFX = "gfx_arch"

class AMDVariantPlugin:
    """
    The AMD ROCm Variant Provider Plugin.
    This class implements the interface expected by the WheelNext standard.
    """
    namespace = "amd"
    # WheelNext static plugin API
    dynamic = False

    @staticmethod
    def _parse_list_env(env_val: str | None) -> list[str]:
        if not envval:
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

    def get_supported_configs(
        self, known_properties: frozenset[VariantPropertyType] | None
    ) -> list[VariantFeatureConfig]:
        """
        This is the standardized method that `pip` will call.
        """
        logger.info(f"[{self.namespace}-variant-provider] Running system detection.")

        configs: list[VariantFeatureConfig] = []

        # TODO: Prioritized list of GFX archs.
        # E.g., dGPU might be preferred over iGPU; PCIe vs. APU.
        env_gfx = os.environ.get("AMD_VARIANT_PROVIDER_GFX_ARCH")
        if not env_gfx:
            gfx_names = self._system_info.get("gfx_names")
        else:
            gfx_names = _parse_list_env(env_gfx);

        # Priority 1: GFX architecture (most specific)
        if gfx_names:
            # The list of all detected GFX architectures is provided.
            # The dependency resolver will try to match them.
            configs.append(
                VariantFeatureConfig(name=AMDVariantFeatureKey.GFX, values=gfx_names)
            )
        # Priority 2: ROCm version (more general)
        # Type `str`
        rocm_version = os.environ.get("AMD_VARIANT_PROVIDER_ROCM_VERSION")
        if not rocm_version:
            # Type `tuple[int, int]`
            rocm_version = self._system_info.get("rocm_version")
        if rocm_version:
            if isinstance(rocm_version, tuple):
                major, minor = rocm_version
                configs.append(
                    VariantFeatureConfig(name=AMDVariantFeatureKey.ROCm, values=[f"rocm{major}.{minor}"])
                )
            elif isinstance(rocm_version, str):
                configs.append(
                    VariantFeatureConfig(name=AMDVariantFeatureKey.ROCm, values=[rocm_version])
                )

        if configs:
            logger.info(f"[{self.namespace}-variant-provider] Detected features: {configs}")
        else:
            logger.warning(f"[{self.namespace}-variant-provider] No AMD features detected.")

        return configs

    def validate_property(self, variant_property: VariantPropertyType) -> bool:
        """
        Validates that a given property is well-formed for the AMD namespace,
        not its existence on the current system.
        """
        assert isinstance(variant_property, VariantPropertyType)
        assert variant_property.namespace == self.namespace

        feature = variant_property.feature
        value = variant_property.value

        if feature == AMDVariantFeatureKey.ROCm:
            # Check if value is like "rocm6.0", "rocm6.1", etc.
            return bool(re.match(r"^rocm\d+\.\d+$", value))
        if feature == AMDVariantFeatureKey.GFX:
            # Check if value is like "gfx90a", "gfx1100", etc.
            return bool(re.match(r"^gfx\d+[0-9a-f]*$", value))

        logger.warning(
            f"Unknown variant feature received for validation: "
            f"`{self.namespace} :: {feature}`.",
        )
        return False

def main() -> int:
+    """Minimal CLI to print detected configs for debugging)."""
+    logging.basicConfig(level=os.environ.get("AMD_VARIANT_PROVIDER_LOGLEVEL", "INFO"))
+    plugin = AMDVariantPlugin()
+    cfgs = plugin.get_supported_configs(None)
+    for c in cfgs:
+        print(f"{plugin.namespace} :: {c.name} :: {', '.join(c.values)}")
+    return 0
+
+if __name__ == "__main__":
+    sys.exit(main())
