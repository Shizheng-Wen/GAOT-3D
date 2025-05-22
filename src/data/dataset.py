"""Utility functions for reading the datasets."""
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Sequence, NamedTuple, Literal
from copy import deepcopy

@dataclass
class Metadata:
  periodic: bool
  group_u: str
  group_c: str
  group_x: str
  type: Literal['gaot']
  fix_x: bool
  domain_x: tuple[Sequence[int], Sequence[int]]
  domain_t: tuple[int, int]
  active_variables: Sequence[int]  # Index of variables in input/output
  chunked_variables: Sequence[int]  # Index of variable groups
  num_variable_chunks: int  # Number of variable chunks
  signed: dict[str, Union[bool, Sequence[bool]]]
  names: dict[str, Sequence[str]]
  global_mean: Sequence[float]
  global_std: Sequence[float]

DATASET_METADATA = {
  'gaot-unstructured/drivaernet_pressure': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([-1.16, -1.20, 0.0], [4.21, 1.19, 1.77]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [False], 'c': [None]},
    names={'u': ['$p$'], 'c': [None]},
    global_mean=[-94.5],
    global_std=[117.25],
  ),
  'gaot-unstructured/drivaernet_shearstress': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([-1.16, -1.20, 0.0], [4.21, 1.19, 1.77]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [False], 'c': [None]},
    names={'u': ['$p$'], 'c': [None]},
    global_mean=[-0.6717,  0.0364, -0.0846],
    global_std=[0.8199, 0.4510, 0.7811],
  )
}

