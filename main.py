import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Subset, Dataset
import subprocess
import threading
import warnings
from datetime import datetime
import time  # 引入时间模块
from torch.utils.tensorboard import SummaryWriter

from utils1.get_rnafm_feature99_new import *
from utils1.model_pair import get_model
from utils1.utils import read_csv, GradualWarmupScheduler

# === 新增导入 precision_score ===
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, precision_score

write_lock = threading.Lock()
results_list = []

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
MAX_WORKERS = 4


def print_log(dataset_name, stage, message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}][{dataset_name}][{stage}] {message}")


def fix_seed(seed):
    if seed is None:
        seed = random.randint(1, 10000)
    torch.set_num_threads(1)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def extract_3mer_rnafm_embedding(sequence, model, alphabet, device):
    batch_converter = alphabet.get_batch_converter()
    data = [("seq1", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12])

    token_embeddings = results["representations"][12][0, 1: len(sequence) + 1, :].cpu().numpy()

    kmers_emb = []
    for i in range(1, len(sequence) - 1):
        kmer_vec = np.mean(token_embeddings[i - 1: i + 2, :], axis=0)
        kmers_emb.append(kmer_vec)

    kmers_emb = np.array(kmers_emb, dtype=np.float32)

    target_len = 99
    if len(kmers_emb) > target_len:
        kmers_emb = kmers_emb[:target_len]
    elif len(kmers_emb) < target_len:
        pad_len = target_len - len(kmers_emb)
        kmers_emb = np.pad(kmers_emb, ((0, pad_len), (0, 0)), 'constant')

    return kmers_emb


def get_rna_prob_matrix(seq, w=101, L=101, cutoff=1e-4):
    pid = os.getpid()
    rnd = np.random.randint(0, 100000)
    name = f"tmp_{pid}_{rnd}"
    clean_seq = seq.replace("N", "A")

    cmd = f'echo {clean_seq} | RNAplfold -W {min(w, len(clean_seq))} -L {min(L, len(clean_seq))} -c {cutoff} --id-prefix {name} --noLP'
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    prob_matrix = np.zeros((101, 101), dtype=np.float32)
    dp_file = f"{name}_0001_dp.ps"
    if os.path.exists(dp_file):
        with open(dp_file, 'r') as f:
            for line in f:
                if 'ubox' in line and 'lbox' in line:
                    pass
                parts = line.split()
                if len(parts) == 4 and parts[3] == 'ubox':
                    i, j = int(parts[0]) - 1, int(parts[1]) - 1
                    p = float(parts[2]) ** 2
                    if i < 101 and j < 101:
                        prob_matrix[i, j] = p
                        prob_matrix[j, i] = p
        os.remove(dp_file)
    if os.path.exists(f"{name}_0001_plp"): os.remove(f"{name}_0001_plp")
    return prob_matrix


class DynamicFeatureDataset(Dataset):
    def __init__(self, sequences, structs, labels, file_name, rnafm_model, alphabet, device):
        self.sequences = sequences
        self.structs = structs
        self.labels = labels
        self.file_name = file_name
        self.rnafm_model = rnafm_model
        self.alphabet = alphabet
        self.device = device
        self.feature_dir = os.path.join("/root/autodl-tmp/lixinyu/lxy/feature_K562_HepG2/pair", file_name)
        os.makedirs(self.feature_dir, exist_ok=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        start_time = time.time()

        rnafm_path = os.path.join(self.feature_dir, f"{idx}_rnafm_3mer.npy")
        struct_path = os.path.join(self.feature_dir, f"{idx}_struct.npy")
        matrix_path = os.path.join(self.feature_dir, f"{idx}_matrix.npy")

        # 1. RNA-FM Feature Extraction
        t0 = time.time()
        rnafm_emb = None
        if os.path.exists(rnafm_path):
            try:
                rnafm_emb = np.load(rnafm_path, allow_pickle=True)
            except:
                pass
        if rnafm_emb is None or not isinstance(rnafm_emb, np.ndarray):
            rnafm_emb = extract_3mer_rnafm_embedding(self.sequences[idx], self.rnafm_model, self.alphabet, self.device)
            np.save(rnafm_path, rnafm_emb)
        time_rnafm = time.time() - t0

        # 2. 1D Structure Processing
        t0 = time.time()
        struct_emb = None
        if os.path.exists(struct_path):
            try:
                struct_emb = np.load(struct_path, allow_pickle=True)
            except:
                pass
        if struct_emb is None:
            struct_emb = process_single_structure(self.structs[idx], target_length=101)
            np.save(struct_path, struct_emb)
        time_struct = time.time() - t0

        # 3. RNAplfold Matrix Generation
        t0 = time.time()
        pair_matrix = None
        if os.path.exists(matrix_path):
            try:
                pair_matrix = np.load(matrix_path, allow_pickle=True)
            except:
                pass
        if pair_matrix is None:
            pair_matrix = get_rna_prob_matrix(self.sequences[idx])
            np.save(matrix_path, pair_matrix)
        time_matrix = time.time() - t0

        # Tensor Conversion
        rnafm_t = torch.tensor(rnafm_emb, dtype=torch.float32).permute(1, 0)
        struct_t = torch.tensor(struct_emb, dtype=torch.float32).unsqueeze(0)
        matrix_t = torch.tensor(pair_matrix, dtype=torch.float32)
        label_t = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)

        total_time = time.time() - start_time

        if idx % 1000 == 0:
            print(f"[Dataset Tracker] Idx {idx} loaded in {total_time:.4f}s | "
                  f"RNA-FM: {time_rnafm:.4f}s | Struct: {time_struct:.4f}s | RNAplfold: {time_matrix:.4f}s")

        return rnafm_t, struct_t, matrix_t, label_t


def collate_fn(batch):
    rnafms, structs, matrices, labels = zip(*batch)
    return (
        torch.stack(rnafms),
        torch.stack(structs),
        torch.stack(matrices),
        torch.stack(labels)
    )


def ensure_2d(x):
    if x.dim() == 1: return x.unsqueeze(-1)
    if x.dim() == 0: return x.view(1, 1)
    return x.view(x.size(0), -1)


def train_simple(model, device, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for rnafm, struct, adj, labels in loader:
        rnafm, struct, adj = rnafm.to(device), struct.to(device), adj.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(rnafm, struct, adj)

        if logits.dim() == 3: logits = logits.squeeze(-1)
        logits = ensure_2d(logits)
        labels = ensure_2d(labels)

        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        all_labels.extend(labels.detach().cpu().numpy().flatten())

    metrics = {}
    if len(all_preds) > 0:
        y_true = np.array(all_labels)
        y_score = np.array(all_preds)
        y_pred = (y_score > 0.5).astype(int)

        metrics['loss'] = total_loss / len(loader)
        try:
            metrics['auc'] = roc_auc_score(y_true, y_score)
        except:
            metrics['auc'] = 0.5
        metrics['acc'] = (y_pred == y_true).mean()
        metrics['aupr'] = average_precision_score(y_true, y_score)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        # === 新增：Precision 计算 ===
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    else:
        metrics = {'loss': 0, 'auc': 0.5, 'acc': 0, 'aupr': 0, 'f1': 0, 'mcc': 0, 'precision': 0}

    return metrics


@torch.no_grad()
def validate_simple(model, device, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for rnafm, struct, adj, labels in loader:
        rnafm, struct, adj = rnafm.to(device), struct.to(device), adj.to(device)
        labels = labels.to(device)

        logits, _ = model(rnafm, struct, adj)

        if logits.dim() == 3: logits = logits.squeeze(-1)
        logits = ensure_2d(logits)
        labels = ensure_2d(labels)

        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    metrics = {}
    if len(all_preds) > 0:
        y_true = np.array(all_labels)
        y_score = np.array(all_preds)
        y_pred = (y_score > 0.5).astype(int)

        metrics['loss'] = total_loss / len(loader)
        try:
            metrics['auc'] = roc_auc_score(y_true, y_score)
        except:
            metrics['auc'] = 0.5
        metrics['acc'] = (y_pred == y_true).mean()
        metrics['aupr'] = average_precision_score(y_true, y_score)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        # === 新增：Precision 计算 ===
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    else:
        metrics = {'loss': 0, 'auc': 0.5, 'acc': 0, 'aupr': 0, 'f1': 0, 'mcc': 0, 'precision': 0}
    return metrics


def run_task_for_dataset(data_file, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}][{data_file}] Task Started on {device}")

    # --- Step 1: Data Loading ---
    t0 = time.time()
    data_path = os.path.join(args.data_path, data_file + '.tsv')
    if not os.path.exists(data_path):
        print(f"Data path {data_path} not found!")
        return
    _, sequences, structs, labels = read_csv(data_path)
    print(f"-> [Step 1] Loading CSV data took {time.time() - t0:.2f}s")

    # --- Step 2: Model Loading ---
    t0 = time.time()
    model_path = os.path.join(args.RNAFM_model_path, "RNA-FM_pretrained.pth")
    rnafm_model, alphabet = load_rnafm_model(model_path, device)
    print(f"-> [Step 2] Loading RNA-FM model took {time.time() - t0:.2f}s")

    # --- Step 3: Dataset Initialization ---
    t0 = time.time()
    ds = DynamicFeatureDataset(sequences, structs, labels, data_file, rnafm_model, alphabet, device)
    print(f"->[Step 3] Initializing Dataset took {time.time() - t0:.2f}s")

    indices = np.arange(len(ds))
    np.random.shuffle(indices)
    split = int(len(ds) * 0.8)

    train_dl = DataLoader(Subset(ds, indices[:split]), batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_dl = DataLoader(Subset(ds, indices[split:]), batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = get_model(args).to(device)

    tr_labels = [labels[i] for i in indices[:split]]
    pos = sum(tr_labels)
    neg = len(tr_labels) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)]).float().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    target_model_path = os.path.join(args.model_save_path, f"{data_file}.pth")
    if os.path.exists(target_model_path):
        print(f"\n-> [INFO] Found existing pre-trained weights at {target_model_path}.")
        print(f"-> [INFO] Loading weights and evaluating directly...")


        model.load_state_dict(torch.load(target_model_path, map_location=device))


        va_m = validate_simple(model, device, test_dl, criterion)

        print(f"[{datetime.now().strftime('%H:%M:%S')}][{data_file}] Evaluation Results: ")
        print(f"   AUC: {va_m['auc']:.4f} | AUPRC: {va_m['aupr']:.4f} | ACC: {va_m['acc']:.4f}")
        print(f"   F1:  {va_m['f1']:.4f} | MCC:   {va_m['mcc']:.4f} | Precision: {va_m['precision']:.4f}")


        with write_lock:
            results_list.append([
                data_file,
                va_m.get('auc', 0),
                va_m.get('acc', 0),
                va_m.get('aupr', 0),
                va_m.get('f1', 0),
                va_m.get('mcc', 0),
                va_m.get('precision', 0)
            ])
        return data_file


    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=300)

    best_auc = 0
    best_metrics = {'auc': 0, 'acc': 0, 'aupr': 0, 'f1': 0, 'mcc': 0, 'precision': 0}
    patience = 0

    print(f"\n-> [Step 4] Starting Epoch Loop...")
    for epoch in range(1, 301):

        t0 = time.time()
        tr_m = train_simple(model, device, train_dl, criterion, optimizer)
        train_time = time.time() - t0


        t1 = time.time()
        va_m = validate_simple(model, device, test_dl, criterion)
        val_time = time.time() - t1

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        if va_m['auc'] > best_auc:
            best_auc = va_m['auc']
            best_metrics = va_m
            patience = 0
            os.makedirs(args.model_save_path, exist_ok=True)
            torch.save(model.state_dict(), target_model_path)
        else:
            patience += 1
            if patience >= args.early_stopping:
                print_log(data_file, "STOP", f"Early stopping. Best AUC: {best_auc:.4f}")
                break

        if epoch % 1 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}][{data_file}] Ep {epoch}: "
                  f"Tr_AUC {tr_m['auc']:.4f} | Va_AUC {va_m['auc']:.4f} | "
                  f"[Time: Train {train_time:.1f}s, Val {val_time:.1f}s]")

    with write_lock:
        results_list.append([
            data_file,
            best_metrics.get('auc', 0),
            best_metrics.get('acc', 0),
            best_metrics.get('aupr', 0),
            best_metrics.get('f1', 0),
            best_metrics.get('mcc', 0),
            best_metrics.get('precision', 0)
        ])

    return data_file


def main(args):
    fix_seed(args.seed)
    data_files = [f.strip() for f in args.data_file.split(',') if f.strip()]
    print(f"Datasets: {data_files}")

    for f in data_files:
        print(f"\n{'=' * 20} Start processing dataset: {f} {'=' * 20}")
        try:
            run_task_for_dataset(f, args)
        except Exception as e:
            import traceback
            traceback.print_exc()


    results_df = pd.DataFrame(results_list, columns=["Dataset", "AUC", "ACC", "AUPRC", "F1", "MCC", "Precision"])
    print("\n=================== FINAL RESULTS ===================")
    print(results_df)

    target_dir = "/root/autodl-tmp/lixinyu/lxy/feature_K562_HepG2"
    os.makedirs(target_dir, exist_ok=True)
    target_excel_path = os.path.join(target_dir, "results5.xlsx")
    results_df.to_excel(target_excel_path, index=False)
    print(f"\nResults saved to: {target_excel_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='MATR3_HepG2', type=str)
    parser.add_argument('--data_path', default='/root/autodl-tmp/lixinyu/lxy/datasets_K562_HepG2', type=str)
    parser.add_argument('--RNAFM_model_path', default='./RNAFM_Model', type=str)
    parser.add_argument('--model_save_path', default='/root/autodl-tmp/lixinyu/lxy/newresults_mine_new1', type=str)
    parser.add_argument('--kmer_channels', default=640, type=int)
    parser.add_argument('--structure_channels', default=1, type=int)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--dynamic_validate', action='store_true')
    parser.add_argument('--seed', default=2026, type=int)
    parser.add_argument('--early_stopping', type=int, default=20)
    parser.add_argument('--conv_channels', type=int, default=128)
    parser.add_argument('--fusion_dim', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--global_dim', type=int, default=640)
    parser.add_argument('--kmer_embed_dim', type=int, default=64)

    args = parser.parse_args()
    main(args)


