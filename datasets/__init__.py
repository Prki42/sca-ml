from .ascad_v1 import ASCADv1FixedKey
from .base import SCADataset

DATASET_CLASSES: dict[str, type[SCADataset]] = {
    "ascad_v1": ASCADv1FixedKey,
}
