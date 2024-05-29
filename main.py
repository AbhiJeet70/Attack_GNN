
from utils.utils import run_experiment

if __name__ == "__main__":
    model_types = ["GCN", "SGC", "GAT", "GIN"]
    data_names = ['cora', 'citeseer', 'polblogs', 'pubmed', 'cora_ml']
    perturbation_rates = [None, 0.05, 0.10, 0.15, 0.20, 0.25, 0.5, 0.75, 1.0]
    attack_methods = ['meta', 'nettack']

data = []
for data_name in data_names:
    for model_type in model_types:
        for attack_method in attack_methods:
            for ptb_rate in perturbation_rates:
                accuracy = run_experiment(model_type=model_type, data_name=data_name, attack_method=attack_method, ptb_rate=ptb_rate)
                data.append([data_name, model_type, attack_method, ptb_rate, accuracy])

# Write data to a CSV file
with open('experiment_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Data Name', 'Model Type', 'Attack Method', 'Perturbation Rate', 'Accuracy'])
    writer.writerows(data)

# Read data from CSV file
df = pd.read_csv('experiment_results.csv')

# Plot data
for data_name in data_names:
    for model_type in model_types:
        for attack_method in attack_methods:
            df_filtered = df[(df['Data Name'] == data_name) & (df['Model Type'] == model_type) & (df['Attack Method'] == attack_method)]
            for ptb_rate in perturbation_rates:
                df_ptb = df_filtered[df_filtered['Perturbation Rate'] == ptb_rate]
                plt.plot(df_ptb['Perturbation Rate'], df_ptb['Accuracy'], label=f"{model_type} - {attack_method}")

plt.xlabel('Perturbation Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Perturbation Rate')
plt.legend()
plt.show()
