import torch
import numpy as np
import rdkit.Chem as Chem
from omegaconf import OmegaConf
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.models import bengio2021flow
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gnn_predictor.mpnn import GraphTransformer, load_mpnn_to_gflow, mol2graph

# TODO: Replace file with argparser maybe
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
        )["sampling_model_state_dict"][0]
    )
)
# Load Algo
algo = TrajectoryBalance(env, ctx, cfg)
# Sample
model.eval()
# on cond_info must pass shape (samples, 32)
# TODO: Figure out how to pass cond info because right now is randn
np.random.seed(42)
torch.manual_seed(42)
samples = algo.create_training_data_from_own_samples(
    model=model, n=10, cond_info=torch.randn(10, 32)
)
trajectories = [sample["traj"] for sample in samples]
valid = [sample["is_valid"] for sample in samples]
rdkit_mols = [ctx.graph_to_obj(traj[-1][0]) for traj in trajectories]
img = Chem.Draw.MolToImageFile(rdkit_mols[1], "test.png")

# Calculate LogP with proxy
smiles = [Chem.MolToSmiles(mol) for mol in rdkit_mols]
MPNN_PROXY = "../../gnn_predictor/model.pth"
proxy_model = load_mpnn_to_gflow(saved_model_path=MPNN_PROXY)
