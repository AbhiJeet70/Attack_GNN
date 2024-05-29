
import torch
import statistics


from models.gnn import GNN
from dataset.dataset import load_data
from dataset.dataloader import get_dataloader

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Main experiment function
def run_experiment(model_type = "GCN", data_name = 'cora', attack_method = 'meta', ptb_rate=None):
    pyg_data = load_data(data_name, attack_method, ptb_rate=ptb_rate)
    data = pyg_data[0].to(device)

    ## PARAMETERS
    input_dim = pyg_data.num_node_features
    hidden_dim = 512
    output_dim = pyg_data.num_classes

    lr=0.01
    weight_decay=0
    epochs = 500
    dropout = 0.5
    patience = 50

    
    model = GNN(model_type, input_dim, hidden_dim, output_dim, epochs, lr, weight_decay, dropout)
    loader = get_dataloader("Neighbor", data, batch_size = 56)
    average = []
    for _ in range(50):
        model.train(loader)
        _, _, acc = model.test(loader)
        average.append(acc)
    
    return statistics.mean(average)
