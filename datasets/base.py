from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class SCADataset(ABC):
    @property
    @abstractmethod
    def input_dim(self) -> int: ...

    @property
    def num_classes(self) -> int:
        return 256

    @property
    @abstractmethod
    def fixed_profiling_key(self) -> bool:
        """Whether all profiling traces share the same key."""
        ...

    @property
    @abstractmethod
    def profiling_key(self) -> Optional[int]:
        """The fixed profiling key byte, or None for random-key datasets."""
        ...

    @abstractmethod
    def leakage(self, plaintext: int, key_guess: int) -> int:
        """Return the target class for a given plaintext byte and key candidate."""
        ...

    @abstractmethod
    def get_profiling(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (traces [N, D] float32, labels [N] int (0 to num_classes-1))."""
        ...

    @abstractmethod
    def get_attack(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Returns (traces [N, D] float32, plaintexts [N] uint8, true_key_byte int)."""
        ...
