from dataclasses import dataclass


@dataclass
class TensorInfo:
    dims: list[int]
    dtype: str
    memory_offset: int
    lifetime_begin: int
    lifetime_end: int
