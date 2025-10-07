import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# for creating causal mask and dataset class
from dataset import BilingualDataset, causal_mask, latest_weights_file_path, get_weights_file_path, get_config
from model import build_transformer


from tokenizers import Tokenizer  # for custom tokenizer
from datasets import load_dataset  # for loading datasets
from tokenizers.models import WordLevel  # for word-level tokenization
# for training word-level tokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace  # for whitespace tokenization


import torchmetrics  # for evaluation metrics
from torch.utils.tensorboard import SummaryWriter  # for tensorboard

import warnings
from tqdm import tqdm
import os
from pathlib import Path


"""
Here we are building greedy decoding function which will be used during inference to generate the target sentence given the source sentence using the trained transformer model and also functions to get dataset, build or load tokenizer and build the transformer model itself.
model - trained transformer model
source - source sentence tensor
source_mask - mask for source sentence
tokenizer_src - tokenizer for source language
tokenizer_tgt - tokenizer for target language
max_len - maximum length of the target sentence to be generated
device - device to run the model on (cpu or gpu)
"""


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(
            1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask,
                           decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


"""
Here we are building the function to run validation on the validation dataset after every epoch during training.
model - trained transformer model
validation_ds - validation dataset dataloader
tokenizer_src - tokenizer for source language
tokenizer_tgt - tokenizer for target language
max_len - maximum length of the target sentence to be generated
device - device to run the model on (cpu or gpu)
print_msg - function to print messages to the console
global_step - current global step of the training
writer - tensorboard writer to log metrics
num_examples - number of examples to run validation on
"""


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()  # set the model to evaluation mode
    count = 0  # counter for number of examples processed

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1  # increment the counter
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(
                device)  # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"  # because we are generating one sentence at a time

            model_out = greedy_decode(
                model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)  # get the model output using greedy decoding

            source_text = batch["src_text"][0]  # get the source text
            target_text = batch["tgt_text"][0]  # get the target text
            model_out_text = tokenizer_tgt.decode(
                model_out.detach().cpu().numpy())  # decode the model output to get the predicted text

            # append the source text to the list
            source_texts.append(source_text)
            expected.append(target_text)  # append the target text to the list
            # append the predicted text to the list
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def get_all_sentences(ds, lang):
    for item in ds:  # iterate through all items in dataset
        yield item[lang]


def get_or_build_tokenizer(config, ds, lang):
    # Path to save/load tokenizer
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):  # if path does not exist then build tokenizer
        print(f"Building {lang} tokenizer...")
        # initialize word-level tokenizer and set unk token for unknown words
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()  # set pre-tokenizer to whitespace
        # splitting words by whitespace and special tokens called UNK, PAD, SOS, EOS which are unknown, padding, start of sentence and end of sentence tokens respectively
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))  # save tokenizer to path
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset(
        f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # building tokenizers for source and target languages
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # 90% train, 10% val split
    # where ds_raw is the entire dataset
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(
        ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt,
                                config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,
                              config['lang_src'], config['lang_tgt'], config['seq_len'])

    # finding the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(
            item['translation'][config['lang_src']]).ids  # getting the token ids for each source and target sentence
        tgt_ids = tokenizer_tgt.encode(
            item['translation'][config['lang_tgt']]).ids  # getting the token ids for each source and target sentence
        # getting the maximum length of source sentence
        max_len_src = max(max_len_src, len(src_ids))
        # getting the maximum length of target sentence
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # creating dataloaders for train and validation datasets
    train_dataloader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    # here we are building the transformer model using the build_transformer function from model.py
    # we are passing the vocab size of source and target languages, sequence length and d_model as parameters to the function
    # and returning the model
    model = build_transformer(vocab_src_len, vocab_tgt_len,
                              config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model


def train_model(config):
    device = "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(
        parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(
        config)
    model = get_model(config, tokenizer_src.get_vocab_size(),
                      tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(
        config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id(
        '[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(
            train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(
                device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(
                device)  # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(
                encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask)  # (B, seq_len, d_model)
            # (B, seq_len, vocab_size)
            proj_output = model.project(decoder_output)

            # Compare the output with the label
            label = batch['label'].to(device)  # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt,
                       config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
