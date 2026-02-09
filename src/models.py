# src/models.py
from dataclasses import dataclass
from typing import List

@dataclass
class Disease:
    name: str
    symptoms: List[str]
    medicines: List[str]

    def __str__(self):
        return self.name
