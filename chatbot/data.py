import os
import json
import pickle
from itertools import chain
from tqdm.auto import tqdm
from torch.utils.data import Dataset

import jsonlines

#from processing import Processing
from preprocess_graph import GNNDataset

class DialoguesDataset(Dataset):
    def __init__(self, prefix, encoder_tokenizer, decoder_tokenizer, args):
        '''self.encoder_x = []
        self.encoder_edge_index = []
        self.encoder_edge_type = []'''

        self.decoder_input_ids = []
        self.labels = []

        #self.txt_encoder_input = []
        self.txt_decoder_input = []
        self.txt_label = []

        #self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

        self._prepare_data(prefix, args)

    def __len__(self):
        #assert len(self.encoder_x) == len(self.decoder_input_ids) == len(self.labels)
        return len(self.decoder_input_ids)

    def __getitem__(self, idx):
        #persona triple, query, response
        #return self.encoder_x[idx], self.encoder_edge_index[idx], self.encoder_edge_type[idx], self.decoder_input_ids[idx], self.labels[idx], self.txt_encoder_input[idx], self.txt_decoder_input[idx], self.txt_label[idx]
        return self.decoder_input_ids[idx], self.labels[idx], self.txt_decoder_input[idx], self.txt_label[idx]

    def _prepare_data(self, prefix, args):
        #read from preprocessed jsonl (triple, persona setntence, query, response)
        f = jsonlines.open(f'{args["dataset_dir"]}/preprocessed/raw/{prefix}.jsonl', 'r')
        data = [inst for inst in f]            

        #create GNNDatset
        #graph_dataset = GNNDataset(root=f'{args["dataset_dir"]}/preprocessed', d_type=prefix, d_len=len(data))

        #iterate jsonl with graph dataset
        for index, line in enumerate(data): 
            #triple = line['triples']
            
            query_tokenized = self.decoder_tokenizer(line['query']).input_ids
            query = line['query']

            response_tokenized = self.decoder_tokenizer(line['response']).input_ids
            response_tokenized += [args['eos_id']]
            response = line['response']
 
            input_ids = (query_tokenized + response_tokenized)
            if len(input_ids) > args['max_len']:
                raise RuntimeError('입출력 길이가 너무 깁니다!!!')
            label_ids = [-100]*(len(input_ids) - len(response_tokenized)) + response_tokenized    
            if(len(input_ids) != len(label_ids)):
                raise RuntimeError('Input 길이와 Label 길이가 다릅니다!!!')
            
            '''self.encoder_x.append(graph_dataset[index].x.tolist())
            self.encoder_edge_index.append(graph_dataset[index].edge_index.tolist())
            self.encoder_edge_type.append(graph_dataset[index].edge_attr.tolist())
            self.txt_encoder_input.append(triple)'''

            self.decoder_input_ids.append(input_ids)
            self.txt_decoder_input.append(query)

            self.labels.append(label_ids)
            self.txt_label.append(response)