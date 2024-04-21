from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetResult:
    metrics: dict[str, Any]
    dataset: str
