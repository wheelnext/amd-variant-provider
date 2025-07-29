from variantlib.models.provider import VariantFeatureConfig

class AMDVariantPlugin:
    namespace = "amd"

    def validate_property(self, variant_property: VariantPropertyType) -> bool:
        pass

    def get_supported_configs(self, known_properties: frozenset[VariantPropertyType] | None) -> list[VariantFeatureConfigType]:
        pass
