import os
import math
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup

from data import DialoguesDataset
from torch_geometric.loader import DataLoader as GraphDataLoader
from preprocess_graph import GNNDataset
from utils import PadCollate


class Trainer:
    def __init__(self, encoder_model, decoder_model, encoder_tokenizer, decoder_tokenizer, args):
        print('Loading the optimizer...')
        self.encoder_optimizer = AdamW(encoder_model.parameters(), lr=args['lr'])
        self.decoder_optimizer = AdamW(decoder_model.parameters(), lr=args['lr'])
        self.best_loss = 1e+10
        self.last_epoch = 0
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

        print('Loading train & valid data...')
        #Loading Graph Dataset
        train_graph_dataset = GNNDataset(root=f'{args["dataset_dir"]}/preprocessed', d_type='train', d_len=100)
        valid_graph_dataset = GNNDataset(root=f'{args["dataset_dir"]}/preprocessed', d_type='valid', d_len=99)
        self.train_graph_loader = GraphDataLoader(train_graph_dataset, batch_size=args["batch_size"], shuffle=False)
        self.valid_graph_loader = GraphDataLoader(valid_graph_dataset, batch_size=args["batch_size"], shuffle=False)

        #Loading Full Dataset
        train_dataset = DialoguesDataset('train', encoder_tokenizer, decoder_tokenizer, args)
        valid_dataset = DialoguesDataset('valid', encoder_tokenizer, decoder_tokenizer, args)
        pad = PadCollate(args)

        self.train_loader = DataLoader(train_dataset,
                                       collate_fn=pad,
                                       shuffle=False,
                                       batch_size=args['batch_size'],
                                       num_workers=1,
                                       pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset,
                                       collate_fn=pad,
                                       batch_size=args['batch_size'],
                                       num_workers=1,
                                       pin_memory=True)

        if not os.path.exists(args['models_dir']):
            os.makedirs(args['models_dir'])

        # Calculate total training steps
        num_batches = len(self.train_loader)
        total_train_steps = args['num_epochs'] * num_batches
        warmup_steps = int(args['warmup_ratio'] * total_train_steps)

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.args = args
        '''self.scheduler = get_polynomial_decay_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
            power=2
        )'''

        if args['checkpoint']:
            self._load_checkpoint()

    def train(self):
        print('Launch training...')

        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, start_epoch + self.args['num_epochs']):
            print('-' * 50 + f'\nEpoch: {epoch}\n' + '-' * 50)

            self.encoder_model.train()
            self.decoder_model.train()
            train_losses = []
            train_perplexity = []

            for batch, graph_batch in tqdm(zip(self.train_loader, self.train_graph_loader)):
                decoder_input_ids, labels, txt_decoder_input_ids, txt_labels = batch
                decoder_input_ids = decoder_input_ids.to(self.args['device'])
                labels = labels.to(self.args['device'])
                graph_batch = graph_batch.to(self.args['device'])

                print(graph_batch.x)
                print(graph_batch.x.shape)
                print("============")
                print(graph_batch.edge_index)
                print(graph_batch.edge_index.shape)
                print("============")
                print(graph_batch.edge_attr)
                print(graph_batch.edge_attr.shape)
                print("============")

                encoder_hidden_states = self.encoder_model(
                    x=graph_batch.x,
                    edge_index=graph_batch.edge_index,
                    #edge_type=graph_batch.edge_index.to_sparse(), 
                    edge_type=graph_batch.edge_attr, 
                )
                outputs = self.decoder_model(
                    input_ids=decoder_input_ids,
                    labels=labels, 
                    encoder_hidden_states=encoder_hidden_states
                )

                loss, logits = outputs[0], outputs[1]

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                #self.scheduler.step()

                train_losses.append(loss.detach())
                ppx = torch.exp(loss.detach())
                train_perplexity.append(ppx)

            train_losses = [loss.item() for loss in train_losses]
            train_perplexity = [ppx.item() if not math.isinf(ppx.item()) else 1e+8 for ppx in train_perplexity]
            train_loss = np.mean(train_losses)
            train_ppx = np.mean(train_perplexity)
            print(f'Train loss: {train_loss} \nTrain perplexity: {train_ppx}')

            self.last_epoch += 1

            valid_loss, valid_ppx = self.validate()

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                state_dict = {
                    'encoder_model_state_dict': self.encoder_model.state_dict(),
                    'decoder_model_state_dict': self.decoder_model.state_dict(),
                    'encoder_optim_state_dict': self.encoder_optimizer.state_dict(),
                    'decoder_optim_state_dict': self.decoder_optimizer.state_dict(),
                    'loss': self.best_loss,
                    'epoch': self.last_epoch
                }

                filename = f"{self.args['model_dir']}/model_best_{round(self.best_loss, 4)}.h5"
                torch.save(state_dict, filename)
                print(f'Checkpoint saved: {filename}')

            print(f'Best valid loss: {self.best_loss}')
            print(f'Valid loss: {valid_loss} \nValid perplexity: {valid_ppx}')

        print('Training completed')

    def validate(self):
        print('Launch validation...')
        self.encoder_model.eval()
        self.decoder_model.eval()

        valid_losses = []
        valid_ppxs = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader)):
                encoder_x, encoder_edge_index, encoder_edge_type, decoder_input_ids, labels, txt_encoder_input_ids, txt_decoder_input_ids, txt_labels = batch
                encoder_x = encoder_x.to(self.args['device'])
                encoder_edge_index = encoder_edge_index.to(self.args['device'])
                encoder_edge_type = encoder_edge_type.to(self.args['device'])
                decoder_input_ids = decoder_input_ids.to(self.args['device'])
                labels = labels.to(self.args['device'])

                #input_ids = bert tokenized graph input
                #decoder_inputs = gpt2 tokenized query
                #labels = gpt2 tokenized response
                encoder_hidden_states = self.encoder_model(
                    x=encoder_x,
                    edge_index=encoder_edge_index,
                    edge_type=encoder_edge_type, 
                )
                outputs = self.decoder_model(
                    input_ids=decoder_input_ids,
                    labels=labels, 
                    encoder_hidden_states=encoder_hidden_states
                )

                loss, logits = outputs[0], outputs[1]

                valid_losses.append(loss.detach())
                ppx = torch.exp(loss.detach())
                valid_ppxs.append(ppx)

            valid_losses = [loss.item() for loss in valid_losses]
            valid_ppxs = [ppx.item() if not math.isinf(ppx.item()) else 1e+8 for ppx in valid_ppxs]
            valid_loss = np.mean(valid_losses)
            valid_ppx = np.mean(valid_ppxs)

            if math.isnan(valid_ppx):
                valid_ppx = 1e+8

        return valid_loss, valid_ppx

    def _load_checkpoint(self):
        path = self.args['checkpoint']
        if os.path.exists(path):
            print('Loading checkpoint...')
            checkpoint = torch.load(path, map_location=self.args['device'])
            self.encoder_model.load_state_dict(checkpoint['encoder_model_state_dict'])
            self.decoder_model.load_state_dict(checkpoint['decoder_model_state_dict'])

            print(f'The training restarts with the specified checkpoint: {os.path.basename(path)}')
            self.encoder_optimizer.load_state_dict(checkpoint['encoder_optim_state_dict'])
            self.decoder_optimizer.load_state_dict(checkpoint['decoder_optim_state_dict'])
            self.best_loss = checkpoint['loss']
            self.last_epoch = checkpoint['epoch']
        else:
            print("Can't find the specified checkpoint")