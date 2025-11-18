"""Configuration helpers for LinearNexus layers and models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T", bound="ConfigBase")


@dataclass
class ConfigBase:
    """Simple dataclass mixin with serialization helpers."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        return cls(**data)
