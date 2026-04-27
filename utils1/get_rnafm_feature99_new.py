

import os
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import gc
import fm  # RNA-FM库


def generate_kmers(sequence, k=3):

    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]


def get_kmer_embedding(kmers, model, alphabet, device, batch_size=512):

    model.eval()
    embeddings = []
    batch_converter = alphabet.get_batch_converter()
    data = [("kmer{}".format(i), kmer) for i, kmer in enumerate(kmers)]

    num_batches = math.ceil(len(data) / batch_size)
    progress_bar = tqdm(total=num_batches, desc="Processing k-mers (k=3)",
                        unit="batch", leave=False, dynamic_ncols=True)

    for i in range(0, len(data), batch_size):
        progress_bar.update(1)
        batch_data = data[i:i + batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)
        seq_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12])
            token_embeddings = results["representations"][12]

        for j in range(len(batch_data)):
            seq_len = seq_lens[j].item()
            seq_emb = token_embeddings[j, 1:seq_len - 1]

            if seq_emb.size(0) >= 3:
                center_emb = seq_emb[1]
            else:
                center_emb = seq_emb.mean(dim=0)

            embeddings.append(center_emb.cpu().numpy())

        del batch_tokens, results, token_embeddings
        torch.cuda.empty_cache()
        gc.collect()

    progress_bar.close()

    kmer_embeddings = np.array(embeddings)
    print(f"k-mer feature shap: {kmer_embeddings.shape}")
    return kmer_embeddings


import numpy as np

def process_single_structure(struct, target_length=101):

    if isinstance(struct, list):

        try:
            probs = np.array(struct, dtype=float)
        except ValueError:

            print(f"Error converting list to float array: {struct[:5]}...")
            return np.zeros(target_length)


    elif isinstance(struct, str):

        struct = struct.strip().replace('[', '').replace(']', '')

        parts = struct.split(',')

        probs = np.array([float(x) for x in parts if x.strip() != ''])


    elif isinstance(struct, np.ndarray):
        probs = struct.astype(float)

    else:
        raise TypeError(f"Unsupported input type for structure: {type(struct)}")

    current_len = len(probs)
    final_emb = np.zeros(target_length)


    limit = min(current_len, target_length)
    final_emb[:limit] = probs[:limit]

    return final_emb

def extract_sequence_features(sequences, model_path=None, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_path is None:
        model_path = "/root/autodl-tmp/lixinyu/lxy/RNA-FM_pretrained.pth"


    print(f"Loading RNA-FM model from: {model_path}")
    model, alphabet = fm.pretrained.rna_fm_t12(model_path)
    model = model.to(device)
    model.eval()

    all_features = []
    batch_converter = alphabet.get_batch_converter()

    print(f"Processing {len(sequences)} sequences...")


    batch_size = 8
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_sequences = sequences[i:i + batch_size]
        batch_data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_sequences)]

        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12])

            token_embeddings = results["representations"][12]


            sequence_mask = (batch_tokens != alphabet.padding_idx)
            sequence_lengths = sequence_mask.sum(dim=1, keepdim=True)
            sequence_representations = (token_embeddings * sequence_mask.unsqueeze(-1)).sum(dim=1) / sequence_lengths

            all_features.append(sequence_representations.cpu().numpy())


        del batch_tokens, results, token_embeddings
        torch.cuda.empty_cache()
        gc.collect()


    all_features = np.vstack(all_features)
    print(f"feature shape: {all_features.shape}")

    return all_features


def extract_kmer_features(sequences, model_path=None, device=None, k=3):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_path is None:
        model_path = "/root/autodl-tmp/lixinyu/lxy/RNA-FM_pretrained.pth"


    print(f"Loading RNA-FM model from: {model_path}")
    model, alphabet = fm.pretrained.rna_fm_t12(model_path)
    model = model.to(device)
    model.eval()

    all_kmer_features = []

    print(f"Processing {len(sequences)} sequences for k-mer features (k={k})...")

    for i, sequence in enumerate(tqdm(sequences)):

        kmers = generate_kmers(sequence, k=k)


        kmer_features = get_kmer_embedding(kmers, model, alphabet, device)
        all_kmer_features.append(kmer_features)

    return all_kmer_features


def load_rnafm_model(model_path=None, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_path is None:
        model_path = "/root/autodl-tmp/lixinyu/lxy/RNA-FM_pretrained.pth"

    print(f"Loading RNA-FM model from: {model_path}")
    model, alphabet = fm.pretrained.rna_fm_t12(model_path)
    model = model.to(device)
    model.eval()
    print("RNA-FM model loaded successfully")
    return model, alphabet


if __name__ == "__main__":
    test_sequences = [
        "AUCG" * 25,
        "GCAU" * 25
    ]

    print("Testing feature extraction...")
    features = extract_sequence_features(test_sequences)
    print(f"Features shape: {features.shape}")

    print("Testing k-mer feature extraction...")
    kmer_features = extract_kmer_features(test_sequences, k=3)
    print(f"Number of sequences with k-mer features: {len(kmer_features)}")
    print(f"First sequence k-mer features shape: {kmer_features[0].shape}")


