from variantlib.models.provider import VariantFeatureConfig

class AMDVariantPlugin:
    namespace = "amd"

    def get_all_configs(self) -> list[VariantFeatureConfig]:
        pass

    def get_supported_configs(self) -> list[VariantFeatureConfig]:
        pass
