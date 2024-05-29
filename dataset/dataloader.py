import math

from torch_geometric.loader import ClusterLoader, ClusterData, NeighborLoader, DataLoader, RandomNodeLoader, GraphSAINTRandomWalkSampler

def get_dataloader(method, data, batch_size, root_nodes = 10):
    # setting up loaders for individual training methods 
    if method == "Random":
        num_parts = math.floor(data.train_mask.numel() / batch_size)
        loader = RandomNodeLoader(data, 
                                  num_parts=num_parts, 
                                  drop_last=True,
                                  shuffle=True)
        return loader

    if method == "Neighbor":
        loader = NeighborLoader(data, 
                                num_neighbors=[batch_size] * 2, 
                                batch_size = root_nodes, #batch size = number of “root” nodes
                                drop_last=True,
                                shuffle=True
                )

        return loader

    if method == "Cluster" :
        num_parts = math.floor(data.train_mask.numel() / batch_size)
        cluster_data = ClusterData(data, num_parts)

        loader = ClusterLoader(cluster_data, 
                                batch_size= root_nodes, #batch size = number of clusters per batch
                                drop_last=True,
                                shuffle=True)

        return loader

    else:
        print("Error: No valid method selected")