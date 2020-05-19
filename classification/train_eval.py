import time
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import sklearn
import numpy as np

def run(model, train_loader, test_loader, epochs, optimizer, scheduler, writer,
        device):

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs/clsf', current_time)
    tsbwriter = SummaryWriter(log_dir+'-raw_seq16_wd0.01')

    train_losses, test_losses = [], []
    best_test_acc = float('inf')

    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss, train_acc = train(model, optimizer, train_loader, device)
        t_duration = time.time() - t
        test_loss, test_acc, clsf_rpt = test(model, test_loader, device)
        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            't_duration': t_duration,
            'clsf_rpt': clsf_rpt,
        }

        writer.print_info(info)
        print('Train Acc: {:.4f}, Test Acc: {:.4f}'.format(info['train_acc'], info['test_acc']))

        tsbwriter.add_scalar('data/train_loss', train_loss, epoch)
        tsbwriter.add_scalar('data/test_loss', test_loss, epoch)
        tsbwriter.add_scalar('data/train_acc', train_acc, epoch)
        tsbwriter.add_scalar('data/test_acc', test_acc, epoch)
        print(clsf_rpt)

        if test_acc < best_test_acc :
            writer.save_checkpoint(model, optimizer, scheduler, epoch)
            best_test_acc = test_acc
        if epoch == epochs or epoch % 100 == 0:
            writer.save_checkpoint(model, optimizer, scheduler, epoch)

    tsbwriter.close()


def train(model, optimizer, loader, device):
    model.train()
    crt = 0
    total_loss = 0
    total_len = 0
    for data in loader:
        optimizer.zero_grad()
        x = data.to(device)
        out = model(x)
        y = torch.reshape(data.y, (data.num_graphs, 2))
        y = torch.argmax(y, 1)
        loss = F.cross_entropy(out, y)
        loss.backward()
        total_loss += data.num_graphs * loss.item()
        crt += y.eq(out.data.max(1)[1]).sum().item()
        total_len += data.num_graphs
        optimizer.step()
    return total_loss / total_len, crt / total_len


def test(model, loader, device):
    model.eval()
    crt = 0
    total_loss = 0
    total_len = 0
    preds = []
    labels = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.to(device)
            out = model(x)
            y = torch.reshape(data.y, (data.num_graphs, 2))
            y = torch.argmax(y, 1)
            loss = F.cross_entropy(out, y)
            total_loss += data.num_graphs * loss.item()
            crt += y.eq(out.data.max(1)[1]).sum().item()
            preds.append(out.data.max(1)[1].detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
            total_len += data.num_graphs

    clsf_rpt = sklearn.metrics.classification_report(np.concatenate(labels, axis=None), \
        np.concatenate(preds, axis=None))
    return total_loss / total_len, crt / total_len, clsf_rpt

