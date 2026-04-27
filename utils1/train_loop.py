
from __future__ import print_function
from tqdm import tqdm
import numpy as np
import torch
import utils.metrics as metrics


def train(model, device, train_loader, criterion, optimizer, batch_size):
    model.train()
    met = metrics.MLMetrics(objective='binary')


    for batch_idx, (kmer_data, whole_data, struct_data, target) in enumerate(train_loader):


        kmer_data = kmer_data.float().to(device)
        whole_data = whole_data.float().to(device)
        struct_data = struct_data.float().to(device)
        target = target.to(device).float()


        if target.dim() > 1 and target.shape[1] > 1:
            target = target.max(dim=1)[0]
        target = target.view(-1, 1)


        if target.shape[0] < 2:
            continue

        optimizer.zero_grad()


        output = model(kmer_data, whole_data, struct_data)

        loss = criterion(output, target)
        prob = torch.sigmoid(output)


        y_np = target.to(device='cpu', dtype=torch.long).detach().numpy()
        p_np = prob.to(device='cpu').detach().numpy()
        met.update(y_np, p_np, [loss.item()])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    return met


def validate(model, device, test_loader, criterion):
    model.eval()
    y_all = []
    p_all = []
    l_all = []

    with torch.no_grad():

        for batch_idx, (kmer_data, whole_data, struct_data, target) in enumerate(test_loader):


            kmer_data = kmer_data.float().to(device)
            whole_data = whole_data.float().to(device)
            struct_data = struct_data.float().to(device)
            target = target.to(device).float()


            if target.dim() > 1 and target.shape[1] > 1:
                target = target.max(dim=1)[0]
            target = target.view(-1, 1)


            output = model(kmer_data, whole_data, struct_data)

            loss = criterion(output, target)
            prob = torch.sigmoid(output)

            y_np = target.to(device='cpu', dtype=torch.long).numpy()
            p_np = prob.to(device='cpu').numpy()
            l_np = loss.item()

            y_all.append(y_np)
            p_all.append(p_np)
            l_all.append(l_np)


    if len(y_all) > 0:
        y_all = np.concatenate(y_all)
        p_all = np.concatenate(p_all)
        l_all = np.array(l_all)

        met = metrics.MLMetrics(objective='binary')
        met.update(y_all, p_all, [l_all.mean()])
        return met, y_all, p_all
    else:

        return metrics.MLMetrics(objective='binary'), [], []