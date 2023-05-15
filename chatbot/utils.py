import numpy as np
import random

import nltk
from nltk.stem import WordNetLemmatizer

import torch
from torch.nn.utils.rnn import pad_sequence


class PadCollate:
    def __init__(self, args):
        self.args = args

    def __call__(self, batch):
        eos_id = self.args['eos_id']
        #encoder_x, encoder_edge_index, encoder_edge_type, decoder_input_ids, labels, txt_encoder_input_ids, txt_decoder_input_ids, txt_labels = [], [], [], [], [], [], [], []
        decoder_input_ids, labels, txt_decoder_input_ids, txt_labels = [], [], [], []

        for idx, seqs in enumerate(batch):
            decoder_input_ids.append(torch.LongTensor(seqs[0]))
            labels.append(torch.LongTensor(seqs[1]))
            #txt_encoder_input_ids.append(seqs[5])
            txt_decoder_input_ids.append(seqs[2])
            txt_labels.append(seqs[3])

        decoder_input_ids = pad_sequence(decoder_input_ids, batch_first=True, padding_value=eos_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        #return encoder_x, encoder_edge_index, encoder_edge_type, decoder_input_ids, labels, txt_encoder_input_ids, txt_decoder_input_ids, txt_labels
        return decoder_input_ids, labels, txt_decoder_input_ids, txt_labels


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def top_k_filter(logits, top_k=0., threshold=-float('Inf'), filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def lemma_sentence(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = ['stop', 'the', 'to', 'and', 'a', 'in', 'it', '\'s', 'is', 'I', 'that', 'had', 'on', 'for', 'were', 'was']
    tokenization = [word for word in nltk.word_tokenize(text) if word not in stop_words]
    sentence = ' '.join([lemmatizer.lemmatize(word) for word in tokenization])
    return sentence
