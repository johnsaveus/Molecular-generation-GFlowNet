import torch
import yaml
from omegaconf import OmegaConf
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.models import bengio2021flow
from gflownet.models.config import ModelConfig
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.algo.config import AlgoConfig
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext

# Load Yaml
yaml_file = "gflownet/tasks/logs/debug_run_seh_frag_2024-11-27_16-03-44/config.yaml"
cfg = OmegaConf.load(yaml_file)
# Load env
env = GraphBuildingEnv()
ctx = FragMolBuildingEnvContext(
    max_frags=cfg.algo.max_nodes,
    num_cond_dim=cfg.cond.temperature.num_thermometer_dim,
    fragments=bengio2021flow.FRAGMENTS,
)
# Load GFN Model
model = GraphTransformerGFN(
    env_ctx=ctx,
    cfg=cfg,
    num_graph_out=cfg.algo.tb.do_predict_n + 1,
    do_bck=cfg.algo.tb.do_parameterize_p_b,
)
model.load_state_dict(
    (
        torch.load(
            "gflownet/tasks/logs/debug_run_seh_frag_2024-11-27_16-03-44/model_state.pt"
        )["models_state_dict"][0]
    )
)
# Load Algo
algo = TrajectoryBalance(env, ctx, cfg)
# Sample
model.eval()
algo.create_training_data_from_own_samples(model, 32)
