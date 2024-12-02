import torch
import os.path as osp
import numpy as np
import rdkit.Chem as Chem
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import DataLoader
from omegaconf import OmegaConf
import torch_geometric.data as gd
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.models import bengio2021flow
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.utils.conditioning import TemperatureConditional
from gnn_predictor.mpnn import GraphTransformer, load_mpnn_to_gflow, mol2graph

# TODO: Replace file with argparser maybe
yaml_file = "config.yaml"
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
model.load_state_dict((torch.load("model_state.pt")["sampling_model_state_dict"][0]))
model.eval()
# Load Algo
algo = TrajectoryBalance(env, ctx, cfg)
# Load cond_info
temp_cond = TemperatureConditional(cfg)
cond_info = temp_cond.sample(400)["encoding"]
# # Sample
# # on cond_info must pass shape (samples, 32)
# # TODO: Figure out how to pass cond info because right now is randn
np.random.seed(42)
torch.manual_seed(42)
samples = algo.create_training_data_from_own_samples(
    model=model, n=400, cond_info=cond_info
)
trajectories = [sample["traj"] for sample in samples]
valid = [sample["is_valid"] for sample in samples]
rdkit_mols = [ctx.graph_to_obj(traj[-1][0]) for traj in trajectories]
# img = Chem.Draw.MolToImageFile(rdkit_mols[1], "test.png")

# Calculate LogP with proxy
smiles = [Chem.MolToSmiles(mol) for mol in rdkit_mols]
pyg_graphs = [mol2graph(mol) for mol in rdkit_mols]
batch = gd.Batch.from_data_list([i for i in pyg_graphs if i is not None])
MPNN_PROXY = "gnn_predictor/model.pth"
proxy_model = load_mpnn_to_gflow(saved_model_path=MPNN_PROXY)

proxy_model.eval()
logp_values = [
    value[0]
    for value in proxy_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    .detach()
    .tolist()
]
# Draw density plot
plt.figure(figsize=(10, 6))
sns.histplot(logp_values, kde=True, stat="density", linewidth=0)
sns.kdeplot(logp_values, color="red", linewidth=2)

plt.title("Density Plot", fontsize=16)
plt.xlabel("Value", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

plt.savefig("density_plot.png")
