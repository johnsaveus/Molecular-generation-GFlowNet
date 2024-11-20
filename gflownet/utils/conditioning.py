import abc
from typing import Dict, Generic, TypeVar

import numpy as np
import torch
from scipy import stats
from torch import Tensor
from torch_geometric import data as gd

from gflownet import LogScalar
from gflownet.config import Config
from gflownet.utils.misc import get_worker_rng
from gflownet.utils.transforms import thermometer

Tin = TypeVar("Tin")
Tout = TypeVar("Tout")


class Conditional(abc.ABC, Generic[Tin, Tout]):
    def sample(self, n):
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self, cond_info: Dict[str, Tensor], data: Tin) -> Tout:
        raise NotImplementedError()

    def encoding_size(self):
        raise NotImplementedError()

    def encode(self, conditional: Tensor) -> Tensor:
        raise NotImplementedError()


class TemperatureConditional(Conditional[LogScalar, LogScalar]):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        tmp_cfg = self.cfg.cond.temperature
        self.upper_bound = 1024
        if tmp_cfg.sample_dist == "gamma":
            loc, scale = tmp_cfg.dist_params
            self.upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
        elif tmp_cfg.sample_dist == "uniform":
            self.upper_bound = tmp_cfg.dist_params[1]
        elif tmp_cfg.sample_dist == "loguniform":
            self.upper_bound = tmp_cfg.dist_params[1]
        elif tmp_cfg.sample_dist == "beta":
            self.upper_bound = 1

    def encoding_size(self):
        return self.cfg.cond.temperature.num_thermometer_dim

    def sample(self, n):
        cfg = self.cfg.cond.temperature
        beta = None
        rng = get_worker_rng()
        if cfg.sample_dist == "constant":
            if isinstance(cfg.dist_params[0], (float, int, np.int64, np.int32)):
                beta = np.array(cfg.dist_params[0]).repeat(n).astype(np.float32)
                beta_enc = torch.zeros((n, cfg.num_thermometer_dim))
            else:
                raise ValueError(f"{cfg.dist_params[0]} is not a float)")
        else:
            if cfg.sample_dist == "gamma":
                loc, scale = cfg.dist_params
                beta = rng.gamma(loc, scale, n).astype(np.float32)
            elif cfg.sample_dist == "uniform":
                a, b = float(cfg.dist_params[0]), float(cfg.dist_params[1])
                beta = rng.uniform(a, b, n).astype(np.float32)
            elif cfg.sample_dist == "loguniform":
                low, high = np.log(cfg.dist_params)
                beta = np.exp(rng.uniform(low, high, n).astype(np.float32))
            elif cfg.sample_dist == "beta":
                a, b = float(cfg.dist_params[0]), float(cfg.dist_params[1])
                beta = rng.beta(a, b, n).astype(np.float32)
            beta_enc = thermometer(
                torch.tensor(beta), cfg.num_thermometer_dim, 0, self.upper_bound
            )

        assert len(beta.shape) == 1, f"beta should be a 1D array, got {beta.shape}"
        return {"beta": torch.tensor(beta), "encoding": beta_enc}

    def transform(
        self, cond_info: Dict[str, Tensor], logreward: LogScalar
    ) -> LogScalar:
        assert len(logreward.shape) == len(
            cond_info["beta"].shape
        ), f"dangerous shape mismatch: {logreward.shape} vs {cond_info['beta'].shape}"
        return LogScalar(logreward * cond_info["beta"])

    def encode(self, conditional: Tensor) -> Tensor:
        cfg = self.cfg.cond.temperature
        if cfg.sample_dist == "constant":
            return torch.zeros((conditional.shape[0], cfg.num_thermometer_dim))
        return thermometer(
            torch.tensor(conditional), cfg.num_thermometer_dim, 0, self.upper_bound
        )
