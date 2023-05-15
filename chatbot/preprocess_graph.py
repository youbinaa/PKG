import os
import os.path as osp

import torch
from torch_geometric.data import Dataset, Data

import pandas as pd
import jsonlines
import numpy as np
from tqdm import tqdm

from transformers import BertTokenizer, BertModel, AutoTokenizer

class GNNDataset(Dataset):
    def __init__(self, root, d_type, d_len, transform=None, pre_transform=None):
        """
        root = where the dataset should be stored. 
        the folder is split into raw_dir (downloaded dataset) and processed dir (processed_dir)
        """
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.d_type = d_type 
        self.d_len = d_len

        super(GNNDataset, self).__init__(root, transform=transform, pre_transform=pre_transform)
        #super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """
        if this file exists in raw_dir, the download is not triggered
        """
        
        return f'{self.d_type}.jsonl'

    @property
    def processed_file_names(self):
        """
        if these files are not found in raw_dir, processing in skipped
        """
        return 'not_implemented.pt' #preprocess always

    '''@property
    def num_edge_types(self):
        return self.edge_attr[:, 0].unique().numel()
    '''

    def download(self):
        pass

    def process(self):
        #self.data = pd.read_csv(self.raw_paths[0], header=None)
        f = jsonlines.open(self.raw_paths[0], 'r')
        self.data = [inst for inst in f]
        for index, line in enumerate(tqdm(self.data)): 
        #for index, persona_graph in tqdm(self.data.iterrows(), total = self.data.shape[0]):
            persona_graph = line['triples']
            node_set = set()
            edge_list = []
            g_graph = {}

            for triple in persona_graph:
                trip = triple.split("\t")

                assert len(trip) == 3

                h,r,t = trip[0], trip[1], trip[2]
                node_set.add(h.strip())
                node_set.add(t.strip())
                edge_list.append(r.strip())
                if h in g_graph.keys():
                    g_graph[h]
                    if t in g_graph[h].keys():
                        g_graph[h][t].append(r)
                    else:
                        g_graph[h][t] = [r]    
                else:
                    g_graph[h] = {}
                    g_graph[h][t] = [r]
            self.nodes = list(node_set)
            assert "i" in self.nodes
            self.nodes.remove("i")
            self.nodes.insert(0, "i")
            node2index = {node: i for i, node in enumerate(self.nodes)}

            # Get node features
            node_feats = self._get_node_features(node2index)
            adj_matrix, edges = self._create_graph(g_graph, node2index)
            # Get edge features
            edge_feats = self._get_edge_features(edges)
            # Get adjacency info
            edge_index = self._get_adjacency_info(adj_matrix)

            # Get labels info
            label = None

            #create data object
            data = Data(x=node_feats, 
                        edge_index=edge_index, 
                        edge_attr=edge_feats)
            torch.save(data, os.path.join(self.processed_dir, f'{self.d_type}_{index}.pt'))

    def _get_node_features(self, node2index):
        #print(node2index)
        """
        Returns a matrix / 2d array of the shape [# of Nodes, Node Feature Size]
        """
        nodes_feats = []
        for i, node in enumerate(node2index.keys()):
            assert node2index[node] == i
            token_ids = self.tokenizer.encode(node)
            input_tensor = torch.tensor(token_ids).unsqueeze(0)
            outputs = self.model(input_tensor)
            embedding = outputs.last_hidden_state[0,0,:]
            nodes_feats.append(embedding.detach().numpy())
        all_node_feats = np.asarray(nodes_feats)
        
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _create_graph(self, g_graph, node2index): 
        adj_matrix = []
        edges = []

        for i in range(len(node2index)):
            adj_matrix.append([0 for i in range(len(node2index))])
        for i,node1 in enumerate(node2index.keys()):
            for j,node2 in enumerate(node2index.keys()):
                if node1 in g_graph.keys() and node2 in g_graph[node1].keys():
                    relations = g_graph[node1][node2]
                    rel = '_and_'.join(relations) #{'i': {'fishing': ['has_hobby', 'like_activity']}}

                    adj_matrix[i][j] = 1
                    #edges.append(relations[0])
                    edges.append(rel)

        return adj_matrix, edges

    def _get_edge_features(self, edges):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        edge_feats = []
        for rel in edges:
            token_ids = self.tokenizer.encode(rel)
            input_tensor = torch.tensor(token_ids).unsqueeze(0)
            outputs = self.model(input_tensor)
            embedding = outputs.last_hidden_state[0,0,:]
            edge_feats.append(embedding.detach().numpy())
        all_edge_feats = np.asarray(edge_feats)
        
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, adj_matrix):
        #print(adj_matrix)
        edge_indices = []
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if adj_matrix[i][j]!=0:
                    edge_indices += [[i, j]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)

        #adj_matrix = torch.tensor(adj_matrix)

        #return adj_matrix.to_sparse()
        #print(edge_indices)
        return edge_indices
    
    def len(self):
        return self.d_len
        #return len(self.processed_file_names)
        #return self.data.shape[0]
    
    def get(self, index):
        data = torch.load(os.path.join(self.processed_dir, f'{self.d_type}_{index}.pt'))
        return data

if __name__=="__main__":
    dataset = GNNDataset(root='../data/temp', d_type="train", d_len=2)
    #dataset.preprocess()
    #print(dataset)
    print("-----")
    #print(dataset[0])

    print(dataset[0].edge_index.t())
    #print(dataset[0].edge_index.t())
    print(dataset[0].edge_attr)
    print(dataset[0].edge_attr.shape)
    print("======")
    #num_relations = len(torch.unique(dataset[0].edge_attr))
    #print(num_relations)
    print(dataset.num_features)
    print(dataset.num_edge_features)

    edge_attrs = [data.edge_attr for data in dataset]
    unique_edge_attrs = torch.cat(edge_attrs).unique()
    num_relations = len(unique_edge_attrs)
    