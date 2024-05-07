from dataclasses import dataclass
from typing_extensions import Any


@dataclass
class DatasetResult:
    metrics: dict[str, Any]
    dataset: str
