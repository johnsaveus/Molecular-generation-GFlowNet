from dataclasses import dataclass, field
from typing import List

from gflownet.utils.misc import StrictDataClass


@dataclass
class SEHTaskConfig(StrictDataClass):
    reduced_frag: bool = False


@dataclass
class SEHMOOTaskConfig(StrictDataClass):
    """Config for the SEHMOOTask

    Attributes
    ----------
    n_valid : int
        The number of valid cond_info tensors to sample.
    n_valid_repeats : int
        The number of times to repeat the valid cond_info tensors.
    objectives : List[str]
        The objectives to use for the multi-objective optimization. Should be a subset of ["seh", "qed", "sa", "mw"].
    online_pareto_front : bool
        Whether to calculate the pareto front online.
    """

    n_valid: int = 15
    n_valid_repeats: int = 128
    objectives: List[str] = field(default_factory=lambda: ["seh", "qed", "sa", "mw"])
    log_topk: bool = False
    online_pareto_front: bool = True


@dataclass
class TasksConfig(StrictDataClass):
    seh: SEHTaskConfig = field(default_factory=SEHTaskConfig)
    seh_moo: SEHMOOTaskConfig = field(default_factory=SEHMOOTaskConfig)
