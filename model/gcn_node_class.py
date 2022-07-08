import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

import torch_geometric
from torch_geometric.nn import GCNConv, GATv2Conv, DynamicEdgeConv
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import sys
import os
# setting path

parent_path = os.path.dirname(os.getcwd())
sys.path.append(parent_path)

print(sys.path)

from utils.data_helper import *


# # dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
#
# print()
# print(f'Dataset: {dataset}:')
# print('======================')
# print(f'Number of graphs: {len(dataset)}')
# print(f'Number of features: {dataset.num_features}')
# print(f'Number of classes: {dataset.num_classes}')
#
# data = dataset[0]  # Get the first graph object.
#
# print()
# print(data)
# print('===========================================================================================================')
#
# # Gather some statistics about the graph.
# print(f'Number of nodes: {data.num_nodes}')
# print(f'Number of edges: {data.num_edges}')
# print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
# print(f'Number of training nodes: {data.train_mask.sum()}')
# print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
# print(f'Has isolated nodes: {data.has_isolated_nodes()}')
# print(f'Has self-loops: {data.has_self_loops()}')
# print(f'Is undirected: {data.is_undirected()}')

EPS = np.finfo(np.float32).eps

__all__ = ['NodeClassPredict']

class DataPrep():
    def __init__(self, data_dir):
        self.graphs = create_graphs("TWITCH", data_dir)
        self.n_graphs = len(self.graphs)
        self.count = 0

    def step(self):

        if self.count >= self.n_graphs:
            self.count = 0

        self.prep_data(self.count)
        self.count += 1

    def prep_data(self, index, use_cuda = False):
        # graph_choice = np.random.randint(0, len(self.graphs))


        self.scaler = StandardScaler()

        self.graph1 = self.graphs[index]
        self.num_nodes = nx.number_of_nodes(self.graph1)
        self.graph1 = nx.convert_node_labels_to_integers(self.graph1, first_label=1)

        self.node_com = self.louvain_communities()

        y = np.zeros(self.num_nodes)
        for i, n in enumerate(list(self.graph1.nodes)):
            y[i] = self.graph1.nodes[n]["label"]

        self.unique_classes, class_counts = np.unique(y, return_counts = True)


        # print(np.unique(y, return_counts = True))

        self.num_classes = np.unique(y).shape[0]


        self.prop_dict = []
        for i,c in enumerate(self.unique_classes.tolist()):
            self.prop_dict.append(class_counts[i] / self.num_nodes)
        # print(self.prop_dict)

        if use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        n_metrics = 5 #+ self.num_classes
        x = np.zeros((self.num_nodes, n_metrics))

        for i, n in enumerate(list(self.graph1.nodes)):
            x[i, :] = self.prep_metrics(n)

        x = self.scaler.fit_transform(x)

        self.x = torch.from_numpy(x).float().to(self.device)


        self.num_features = n_metrics

        self.y = torch.from_numpy(y).long().to(self.device)

        self.edge_index = (torch.from_numpy(self.get_edgelist())).long().to(self.device)

    def get_edgelist(self):


        edgelist = nx.to_edgelist(self.graph1)
        edgelist = np.array([[item[0] - 1, item[1] - 1] for item in edgelist]).transpose()

        self.edgelist = edgelist

        return self.edgelist

    def prep_metrics(self, n):
        path_avg, path_std = self.get_paths(n)
        degree = self.graph1.degree[n]
        cluster = nx.clustering(self.graph1, n)
        community = self.node_community(n)
        #class_props = self.prop_dict

        return path_avg, path_std, degree, cluster, community#, *class_props

    def get_paths(self, n):
        paths = nx.single_source_shortest_path_length(self.graph1, n)
        path_keys = list(paths.keys())
        all_paths = []
        for key in path_keys:
            all_paths.append(paths[key])
        all_paths = np.array(all_paths)

        return np.mean(all_paths), np.std(all_paths)

    def louvain_communities(self):
        communities = nx.algorithms.community.louvain_communities(self.graph1, resolution = 0.5)
        node_coms = {}
        for i, set in enumerate(communities):
            for node in set:
                node_coms[node] = i
        # print(node_coms)
        return node_coms

    def node_community(self, n):

        graph_neighbours = [nb for nb in self.graph1.neighbors(n)]
        n_neighbors = len(graph_neighbours)

        if n_neighbors == 0:
            return 0

        neighbourhood_matches = 0
        for n2 in graph_neighbours:
            if self.node_com[n2] == self.node_com[n]:
                neighbourhood_matches += 1




        return neighbourhood_matches / n_neighbors

class NodeFeatPredict(torch.nn.Module):
    def __init__(self, hidden_channels, num_prop, dataset):
        super().__init__()
        torch.manual_seed(12345678)
        self.num_prop = num_prop

        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

        self.conv_inter = GCNConv(hidden_channels, hidden_channels)

        self.out_soft = nn.Softmax(dim  = 0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # print(x)
        # print(edge_index)
        x = F.dropout(x, p = 0.25, training = self.training)

        for i in range(self.num_prop):
            x = self.conv_inter(x, edge_index)
            # x = x.relu()
            x = F.dropout(x, p=0.25, training=self.training)
        x = x.relu()
        x = self.conv2(x, edge_index)
        # x = x.relu()
        x = self.out_soft(x)
        return x

class NodeClassPredict(torch.nn.Module):
    def __init__(self, hidden_channels, num_prop, dataset):
        super().__init__()
        torch.manual_seed(12345678)
        self.num_prop = num_prop

        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

        self.conv_inter = GCNConv(hidden_channels, hidden_channels)

        self.out_soft = nn.Softmax(dim  = 0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # print(x)
        # print(edge_index)
        x = F.dropout(x, p = 0.25, training = self.training)

        for i in range(self.num_prop):
            x = self.conv_inter(x, edge_index)
            # x = x.relu()
            x = F.dropout(x, p=0.25, training=self.training)
        x = x.relu()
        x = self.conv2(x, edge_index)
        # x = x.relu()
        x = self.out_soft(x)
        return x

class NodeClassPredictLight(torch.nn.Module):
    def __init__(self, hidden_channels, num_prop, dataset):
        super().__init__()
        torch.manual_seed(12345678)
        self.num_prop = num_prop

        self.conv1 = GCNConv(dataset.num_features, dataset.num_classes)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
        self.lin   = nn.Linear(hidden_channels, dataset.num_classes)

        self.conv_inter = GCNConv(hidden_channels, hidden_channels)

        self.out_soft = nn.Softmax(dim  = 0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # # print(x)
        # # print(edge_index)
        # x = F.dropout(x, p = 0.25, training = self.training)
        #
        # for i in range(self.num_prop):
        #     x = self.conv_inter(x, edge_index)
        #     # x = x.relu()
        #     x = F.dropout(x, p=0.25, training=self.training)
        x = x.relu()
        # #x = self.lin(x)
        # x = self.conv2(x, edge_index)
        # # x = x.relu()
        # x = self.out_soft(x)
        return x

class NodeClassPredictLinear(torch.nn.Module):
    def __init__(self, hidden_channels, num_prop, dataset):
        super().__init__()
        torch.manual_seed(12345678)
        self.num_prop = num_prop

        self.conv1 = nn.Linear(dataset.num_features, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, dataset.num_classes)

        self.conv_inter = nn.Linear(hidden_channels, hidden_channels)

        self.out_soft = nn.Softmax(dim  = 0)

    def forward(self, x, edge_index):
        x = self.conv1(x)
        # print(x)
        # print(edge_index)
        x = F.dropout(x, p = 0.25, training = self.training)

        # for i in range(self.num_prop):
        #     x = self.conv_inter(x, edge_index)
        #     # x = x.relu()
        #     x = F.dropout(x, p=0.25, training=self.training)
        # x = F.dropout(x, p=0.25, training=self.training)
        x = self.conv_inter(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = x.relu()

        x = self.conv_inter(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = x.relu()

        x = self.conv_inter(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = x.relu()



        #x = F.dropout(x, p=0.25, training=self.training)
        x = self.conv2(x)
        # x = x.relu()
        x = self.out_soft(x)
        return x


def diagnostic_plot(G, pred, true, show = True, index = 1):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(24, 8))

    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.get_shared_y_axes().join(ax1, ax2)

    pos = nx.spring_layout(
        G, k=0.1, iterations=100)

    # colors = []
    #
    # for g in G.nodes:
    #     colors.append(G.nodes[g]["target"])

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=25,
        node_color=true,
        alpha=1,
        linewidths=1,
        ax = ax1)  # ,
    # font_size=1.5)
    nx.draw_networkx_edges(G, pos, alpha=0.75, width=0.5, ax = ax1)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=25,
        node_color=pred,
        alpha=1,
        linewidths=1,
        ax = ax2)  # ,
    # font_size=1.5)
    nx.draw_networkx_edges(G, pos, alpha=0.75, width=0.5, ax = ax2)


    ax3.hist([pred, true], label = ["pred", "true"])

    ax1.set_title("True labels")
    ax2.set_title("Predicted labels")
    ax3.set_title("Label distributions")

    ax3.legend(shadow = True)

    plt.savefig(f"graph_example_{index}_class.jpg")
    if show:
        plt.show()

data = DataPrep('/home/alex/Projects/GRAN_social/data/')
data.step()


model = NodeClassPredictLight(hidden_channels = 30, num_prop=0, dataset = data)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()



device = torch.device("cpu")

model.to(device)

def train():

      try:
        criterion = torch.nn.CrossEntropyLoss(weight = torch.from_numpy(1 / np.array(data.prop_dict)).float())
      except:
        pass
      #



      data.step()

      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.

      loss = criterion(out, data.y)  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()

      all_out = []
      all_y   = []

      for i in range(10):

          data.step()

          out = model(data.x, data.edge_index).cpu().detach().numpy()
          y = data.y.cpu().detach().numpy().tolist()

          out = np.argmax(out, axis=1).tolist()  # Use the class with highest probability.#

          all_out += out
          all_y += y

          diagnostic_plot(data.graph1, out, data.y.cpu().detach().numpy(), show = False, index = i)



      acc = accuracy_score(all_y, all_out)
      f1 = f1_score(all_y, all_out, average="weighted")

      # pred = out.argmax(dim=1)  # Use the class with highest probability.
      # test_correct = pred == data.y  # Check against ground-truth labels.
      # test_acc = int(np.sum(test_correct.cpu().detach().numpy())) / data.x.shape[0]#/ int(np.sum(data))  # Derive ratio of correct predictions.
      return acc, f1

test_acc, f1 = test()
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test f1: {f1:.4f}')

pbar = tqdm(range(1, 400))
for epoch in pbar:
    loss = train()
    pbar.set_description(f"{loss:.4f}")
    # if epoch % 25 == 0:
    #     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

test_acc, f1 = test()
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test f1: {f1:.4f}')