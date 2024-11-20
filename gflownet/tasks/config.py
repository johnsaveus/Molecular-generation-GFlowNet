from dataclasses import dataclass, field
from gflownet.utils.misc import StrictDataClass


@dataclass
class SEHTaskConfig(StrictDataClass):
    reduced_frag: bool = False


@dataclass
class TasksConfig(StrictDataClass):
    seh: SEHTaskConfig = field(default_factory=SEHTaskConfig)
