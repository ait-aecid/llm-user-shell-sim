from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Example:
    text: str
    label: str
    group: Optional[str] = None

    # extra metadata (helpful for debugging + analysis)
    log_type: Optional[str] = None
    path: Optional[str] = None
    line_no: Optional[int] = None


