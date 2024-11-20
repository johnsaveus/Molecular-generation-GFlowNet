from dataclasses import dataclass, field
from typing import Any, List

from gflownet.utils.misc import StrictDataClass


@dataclass
class TempCondConfig(StrictDataClass):
    """Config for the temperature conditional.

    Attributes
    ----------

    sample_dist : str
        The distribution to sample the inverse temperature from. Can be one of:
        - "uniform": uniform distribution
        - "loguniform": log-uniform distribution
        - "gamma": gamma distribution
        - "constant": constant temperature
        - "beta": beta distribution
    dist_params : List[Any]
        The parameters of the temperature distribution. E.g. for the "uniform" distribution, this is the range.
    num_thermometer_dim : int
        The number of thermometer encoding dimensions to use.
    """

    sample_dist: str = "uniform"
    dist_params: List[Any] = field(default_factory=lambda: [0.5, 32])
    num_thermometer_dim: int = 32


@dataclass
class ConditionalsConfig(StrictDataClass):
    valid_sample_cond_info: bool = True
    temperature: TempCondConfig = field(default_factory=TempCondConfig)
