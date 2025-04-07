from typing import TypeVar, Tuple, Union, MutableSequence
T = TypeVar("T")
def make_pair(x:Union[T, Tuple[T,T]])->Tuple[T,T]:
    if isinstance(x, (list,tuple)):
        assert len(x) == 2, f"Expected length 2, got {len(x)}"
        return x
    return x, x

def force_make_pair(x:T)->Tuple[T,T]:
    if isinstance(x, (list, dict)): # mutable
        return x, x.copy()
    return x,x

def is_pair(x:Union[T,Tuple[T,T]])->bool:
    return isinstance(x, (list, tuple)) and len(x) == 2