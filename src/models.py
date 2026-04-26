from dataclasses import dataclass, field
from typing import List

@dataclass
class Disease:
    name: str
    symptoms: List[str]
    medicines: List[str]
    severity: str = "medium"
    urgency: str = "plan"
    advice: List[str] = field(default_factory=list)
    when_to_seek_help: List[str] = field(default_factory=list)

    def __str__(self):
        return self.name
