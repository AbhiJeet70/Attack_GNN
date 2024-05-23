
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from deeprobust.graph.data import Dataset, PrePtbDataset, Dpr2Pyg

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
def load_data(name, attack_method, setting=None, ptb_rate=None):
    if ptb_rate:
        perturbed_data = PrePtbDataset(root='/tmp/', name=name, attack_method=attack_method, ptb_rate=ptb_rate)
        return perturbed_data.adj
    data = Dataset(root='/tmp/', name=name, setting=setting)
    pyg_data = Dpr2Pyg(data)
    return pyg_data