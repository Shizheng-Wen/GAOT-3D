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
  'incompressible_fluids/drivaernet_pressure': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([-1.16, -1.20, 0.0], [4.21, 1.19, 1.77]),
    domain_t=None,
    fix_x=False,
    active_variables=None,
    chunked_variables=None,
    num_variable_chunks=1,
    signed={'u': [False], 'c': [None]},
    names={'u': ['$p$'], 'c': [None]},
    global_mean=[-93.4105],  #-94.5
    global_std=[120.7879], #117.25
  ),
  'incompressible_fluids/drivaernet_shearstress': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([-1.16, -1.20, 0.0], [4.21, 1.19, 1.77]),
    domain_t=None,
    fix_x=False,
    active_variables=None,
    chunked_variables=None,
    num_variable_chunks=1,
    signed={'u': [False], 'c': [None]},
    names={'u': ['$p$'], 'c': [None]},
    global_mean=[-0.6717,  0.0364, -0.0846],
    global_std=[0.8199, 0.4510, 0.7811],
  ),
  'incompressible_fluids/nasa_crm': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='gaot',
    domain_x=([2.3495, -29.460142, 2.3101413], [66.744965, 29.460142,  8.833843]),
    domain_t=None,
    fix_x=False,
    active_variables=None,
    chunked_variables=None,
    num_variable_chunks=1,
    signed={'u': [False,False, False, False], 'c': [False, False]},
    names={'u': ['$p$', 'sfc_x', 'sfc_y', 'sfc_z'], 'c': ['Mach', 'AOA']},
    global_mean=[-3.3177e-02, 1.4710e-03,  6.4260e-06, -2.2570e-06],
    global_std=[0.3108, 0.0010, 0.0005, 0.0007],
  ),
  'incompressible_fluids/nasa_crm_pressure': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='gaot',
    domain_x=([2.3495, -29.460142, 2.3101413], [66.744965, 29.460142,  8.833843]),
    domain_t=None,
    fix_x=False,
    active_variables=None,
    chunked_variables=None,
    num_variable_chunks=1,
    signed={'u': [False], 'c': [False, False]},
    names={'u': ['$p$'], 'c': ['Mach', 'AOA']},
    global_mean=[-3.3177e-02],
    global_std=[0.3108],
  ),
  'incompressible_fluids/nasa_crm_sfc': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='gaot',
    domain_x=([2.3495, -29.460142, 2.3101413], [66.744965, 29.460142,  8.833843]),
    domain_t=None,
    fix_x=False,
    active_variables=None,
    chunked_variables=None,
    num_variable_chunks=1,
    signed={'u': [False, False, False], 'c': [False, False]},
    names={'u': ['sfc_x', 'sfc_y', 'sfc_z'], 'c': ['Mach', 'AOA']},
    global_mean=[1.4710e-03,  6.4260e-06, -2.2570e-06],
    global_std=[0.0010, 0.0005, 0.0007],
  ),
  'incompressible_fluids/drivaerml_pressure': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([-0.943, -1.14, -0.318], [4.14, 1.14, 1.25]),
    domain_t=None,
    fix_x=False,
    active_variables=None,
    chunked_variables=None,
    num_variable_chunks=1,
    signed={'u': [False], 'c': [None]},
    names={'u': ['$p$'], 'c': [None]},
    global_mean=[-0.3046],
    global_std=[0.3560],
  ),
  'incompressible_fluids/drivaerml_wss': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([-0.943, -1.14, -0.318], [4.14, 1.14, 1.25]),
    domain_t=None,
    fix_x=False,
    active_variables=None,
    chunked_variables=None,
    num_variable_chunks=1,
    signed={'u': [False, False, False], 'c': [None]},
    names={'u': [ 'wss_x', 'wss_y', 'wss_z'], 'c': [None]},
    global_mean=[-1.2049,  0.0015, -0.0724],
    global_std=[2.0773, 1.3518, 1.1098],
  ),
  'incompressible_fluids/drivaerml': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='gaot',
    domain_x=([-0.943, -1.14, -0.318], [4.14, 1.14, 1.25]),
    domain_t=None,
    fix_x=False,
    active_variables=None,
    chunked_variables=None,
    num_variable_chunks=1,
    signed={'u': [False, False, False, False], 'c': [None]},
    names={'u': ['$p$', 'wss_x', 'wss_y', 'wss_z'], 'c': [None]},
    global_mean=[-0.3046, -1.2049,  0.0015, -0.0724],
    global_std=[0.3560, 2.0773, 1.3518, 1.1098],
  )
}

