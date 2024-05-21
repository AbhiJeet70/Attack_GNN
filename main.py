
from utils.utils import run_experiment

if __name__ == "__main__":
    perturbation_rates = [None, 0.05, 0.10, 0.15, 0.20, 0.25]
    for ptb_rate in perturbation_rates:
        avg_accuracy = run_experiment(ptb_rate=ptb_rate)
        print(f'Average Accuracy with perturbation rate {ptb_rate}: {avg_accuracy:.4f}')
