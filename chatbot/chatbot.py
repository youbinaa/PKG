import yaml
import torch
import nltk
from glob import glob
from argparse import ArgumentParser
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import EncoderDecoderConfig, EncoderDecoderModel

from utils import set_seed
from custom_gpt import CustomGraphGPT2
from custom_gnn import GAT, RGAT

def main(args):
    set_seed(args['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args['device'] = device

    encoder_tokenizer, decoder_tokenizer = load_tokenizer(args)
    encoder_model, decoder_model = load_model(args, encoder_tokenizer, decoder_tokenizer, device)

    if args['mode'] == 'train':
        from train import Trainer
        trainer = Trainer(encoder_model, decoder_model, encoder_tokenizer, decoder_tokenizer, args)
        trainer.train()
    '''elif args['mode'] == 'interact':
        from interact import Chatbot
        chatbot = Chatbot(model, tokenizer, args)
        chatbot.run()'''


def load_tokenizer(args):
    encoder_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    decoder_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    ''' special_tokens = ['<speaker1>', '<speaker2>']
    tokenizer.add_special_tokens({
        'bos_token': '<bos>',
        'additional_special_tokens': special_tokens
    })

    # add new token ids to args
    special_tokens += ['<bos>', '<eos>']
    sp1_id, sp2_id, bos_id, eos_id = tokenizer.encode(special_tokens)
    args['sp1_id'] = sp1_id
    args['sp2_id'] = sp2_id
    args['bos_id'] = bos_id
    args['eos_id'] = eos_id'''
    args['bos_id'] = decoder_tokenizer.bos_token_id
    args['eos_id'] = decoder_tokenizer.eos_token_id

    return encoder_tokenizer, decoder_tokenizer


def load_model(args, encoder_tokenizer, decoder_tokenizer, device):
    config_encoder = BertConfig()
    config_decoder = GPT2Config.from_pretrained(args['model_name'])
    config_decoder.is_decoder = True
    config_decoder.add_cross_attention = True

    #encoder_model = BertModel.from_pretrained('bert-base-uncased', config = config_encoder).to(device)
    #encoder_model = GAT(in_channels=768, hidden_channels=256, out_channels=768, heads=8).to(device)
    encoder_model = RGAT(in_channels=768, hidden_channels=256, out_channels=768, num_relations=62).to(device) #relations 수정 필요 (blank, none 포함되어있음)
    decoder_model = CustomGraphGPT2.from_pretrained(args['model_name'], config=config_decoder).to(device)

    #model = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model).to(device)
    
    #encoder_model.resize_token_embeddings(len(encoder_tokenizer))
    #decoder_model.resize_token_embeddings(len(decoder_tokenizer))

    return encoder_model, decoder_model


if __name__ == '__main__':
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')

    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, default = "train",
                        help='Pass "train" for training mode and "interact" for interaction mode')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint of the model')

    user_args = parser.parse_args()
    arguments = yaml.safe_load(open('config.yml'))
    arguments.update(vars(user_args))

    main(arguments)
