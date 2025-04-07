import torch
import torch.nn as nn
from typing import Optional, List, Mapping



class BufferList(nn.Module):
    def __init__(self, buffers: Optional[List[torch.Tensor]] = None):
        super().__init__()
        self._length = 0
        if buffers is not None:
            for buffer in buffers:
                self.append(buffer)

    def __setitem__(self, idx: int, value: torch.Tensor):
        assert 0 <= idx < self._length, "Index out of range"
        self.register_buffer(f"_{idx}", value)

    def __getitem__(self, idx: int) -> torch.Tensor:
        assert 0 <= idx < self._length, "Index out of range"
        return getattr(self, f"_{idx}")

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for idx in range(self._length):
            yield getattr(self, f"_{idx}")

    def append(self, buffer: torch.Tensor):
        idx = self._length
        self.register_buffer(f"_{idx}", buffer)
        self._length += 1

    def extend(self, buffers: List[torch.Tensor]):
        for buffer in buffers:
            self.append(buffer)

    def remove(self, buffer: torch.Tensor):
        for idx in range(self._length):
            if getattr(self, f"_{idx}") is buffer:
                self._remove_at_index(idx)
                return
        raise ValueError("Buffer not found in BufferList")

    def pop(self, idx: int = -1) -> torch.Tensor:
        if idx < 0:
            idx += self._length
        assert 0 <= idx < self._length, "Index out of range"
        buffer = getattr(self, f"_{idx}")
        self._remove_at_index(idx)
        return buffer

    def _remove_at_index(self, idx: int):
        for i in range(idx, self._length - 1):
            next_buffer = getattr(self, f"_{i + 1}")
            self.register_buffer(f"_{i}", next_buffer)
        delattr(self, f"_{self._length - 1}")
        self._length -= 1

    def __contains__(self, buffer: torch.Tensor) -> bool:
        for idx in range(self._length):
            if getattr(self, f"_{idx}") is buffer:
                return True
        return False

    def __str__(self) -> str:
        return str([getattr(self, f"_{idx}") for idx in range(self._length)])

    def __repr__(self) -> str:
        return repr([getattr(self, f"_{idx}") for idx in range(self._length)])


class BufferDict(nn.Module):
    def __init__(self, kwargs:Optional[Mapping[str, torch.Tensor]] = None):
        super().__init__()
        if kwargs is not None:
            for key, value in kwargs.items():
                self.register_buffer(key, value)
    def __setitem__(self, key:str, value:torch.Tensor):
        self.register_buffer(key, value)
    def __getitem__(self, key:str)->torch.Tensor:
        return getattr(self, key)
    def __len__(self)->int:
        return len(self._buffers)
    def __iter__(self):
        return iter(self._buffers)
    def __contains__(self, key:str)->bool:
        return key in self._buffers
    def __str__(self)->str:
        return str(self._buffers)
    def __repr__(self)->str:
        return repr(self._buffers)
    def keys(self):
        return self._buffers.keys()
    def values(self):
        return self._buffers.values()
    def items(self):
        return self._buffers.items()
    def update(self, kwargs:Mapping[str, torch.Tensor]):
        for key, value in kwargs.items():
            self.register_buffer(key, value)
    def get(self, key:str, default:Optional[torch.Tensor] = None)->Optional[torch.Tensor]:
        if key in self._buffers:
            return getattr(self, key)
        return None
    def pop(self, key:str)->torch.Tensor:
        value = getattr(self, key)
        delattr(self, key)
        return value
    def asdict(self)->Mapping[str, torch.Tensor]:
        return self._buffers
    @property 
    def default(self)->torch.Tensor:
        return next(iter(self._buffers.values()))
