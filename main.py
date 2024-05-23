
from utils.utils import run_experiment

if __name__ == "__main__":
    model_types = ["GCN","SGC","GAT","GIN"]
    data_names = ["cora","cora-ml","citeseer", "pubmed"]
    perturbation_rates = [None, 0.05, 0.10, 0.15, 0.20, 0.25]

    for name in data_names:
        for model in model_types:
            for ptb_rate in perturbation_rates:
                avg_accuracy = run_experiment(model_type= model, data_name = name, ptb_rate=ptb_rate)
                print(f'Average Accuracy with perturbation rate {ptb_rate}: {avg_accuracy:.4f}')
