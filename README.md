# GNN Attack Simulation

This repository contains the code for simulating attacks on Graph Convolutional Networks (GCNs) using the Cora dataset. It demonstrates how perturbations affect the accuracy of GCNs through a federated learning approach. The project is structured into modular Python files, each dedicated to a different aspect of the simulation.

## Repository Structure
```
Attack_GNN/
├── README.md
├── dataset
│ └── dataset.py
├── GCN
│ └── GCN.py
├── training
│ └── training.py
├── utils
│ └── utils.py
└── main.py
```

- `dataset.py`: Functions to load the Cora dataset, with and without perturbations.
- `GCN.py`: Definition of the GCN model.
- `training.py`: Training and evaluation logic for the GCN.
- `utils.py`: Contains utility functions for running experiments and aggregating results.
- `main.py`: Main script that executes experiments across various perturbation rates.

## Setup

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Torch-Geometric
- Gensim 3.8.3
- DeepRobust
- Git

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Attack_GNN.git
    cd Attack_GNN
    ```

2. Install the required Python packages:
    ```bash
    pip install torch torchvision torchaudio
    pip install torch-geometric
    pip install gensim==3.8.3
    pip install git+https://github.com/DSE-MSU/DeepRobust.git
    ```

### Usage

1. Run the main script to execute the experiments:
    ```bash
    python main.py
    ```

   This script will iterate through various perturbation rates and log the average accuracy of the GCN model at each rate.

### Notes

- Ensure the dataset paths are correctly set in the `dataset.py` script.
- The training process uses CUDA if available; verify your CUDA setup if you intend to use GPU acceleration.

## How It Works

1. **Dataset Loading:** The Cora dataset is loaded with options for applying perturbations.
2. **Model Definition:** A GCN model is defined in `GCN.py`.
3. **Model Training and Evaluation:** The model is trained and evaluated in the `training.py` script, which logs losses and accuracy.
4. **Experimentation:** The `utils.py` script handles running multiple experiments and calculating statistical measures.
5. **Results:** The `main.py` script drives the entire process, reporting the model's performance under various conditions.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
