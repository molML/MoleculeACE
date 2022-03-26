"""
All functions required to train/test the GNNs
Alisa Alenicheva, Jetbrains research, Februari 2022
"""

import os

import numpy as np
from dgllife.model import GCNPredictor, GATPredictor, GINPredictor
from dgllife.utils import Meter, EarlyStopping
import torch
from torch import nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from MoleculeACE.GNN.data import get_moleculecsv_dataset
from MoleculeACE.GNN.models.model import init_model
from MoleculeACE.benchmark.utils import get_torch_device, RANDOM_SEED
import random

from MoleculeACE.benchmark.utils.const import define_default_log_dir, CONFIG_PATH_GENERAL

from MoleculeACE.benchmark.utils import get_config
general_settings = get_config(CONFIG_PATH_GENERAL)

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def train_pipeline(train_loader, val_loader, descriptor, algorithm, epochs, config, result_path, logs_path):
    if logs_path is None:
        logs_path = define_default_log_dir()
    model = init_model(config, algorithm, descriptor)
    device = get_torch_device()
    model = model.to(device)
    # Initiate the loss, optimizer, early stopper and writer that pytorch uses during training
    loss_criterion = nn.SmoothL1Loss(reduction='none')
    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    stopper = EarlyStopping(patience=config['patience'], filename=os.path.join(logs_path, 'model.pth'),
                            metric=general_settings['metric'])
    writer = SummaryWriter(log_dir=result_path)
    # Train the model for n epochs
    model, _, _ = train_model(train_loader, val_loader, model, loss_criterion, optimizer, epochs, stopper, writer,
                              metric=general_settings['metric'])
    return model, stopper


def train_model(train_loader, val_loader, model, loss_criterion, optimizer, epochs, stopper, writer, metric='rmse'):
    for epoch in range(epochs):
        # Train
        run_a_train_epoch(epoch, model, train_loader, loss_criterion, optimizer, writer, metric=metric,
                          num_epochs=epochs)

        # Validation and early stop
        val_rmse, val_r2 = run_an_eval_epoch(model, val_loader)
        early_stop = stopper.step(val_rmse, model)
        print(
            f"epoch {epoch + 1}/{epochs}, validation rmse {val_rmse}, validation r2 {val_r2}, "
            f"best validation rmse {stopper.best_score}")

        writer.add_scalar(f"Val/rmse", val_rmse, epoch)
        writer.add_scalar(f"Val/r2", val_r2, epoch)
        if early_stop:
            break
    return model, val_rmse, val_r2


def run_a_train_epoch(epoch, model, data_loader, loss_criterion, optimizer, writer, num_epochs=1000, print_every=20,
                      metric='rmse'):
    device = get_torch_device()

    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):

        smiles, bg, labels, masks = batch_data
        if len(smiles) == 1:
            # Avoid potential issues with batch normalization
            continue

        labels, masks = labels.to(device), masks.to(device)
        logits = predict_logits(model, bg)
        # Mask non-existing labels
        loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(logits, labels, masks)
        if batch_id % print_every == 0:
            print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
                epoch + 1, num_epochs, batch_id + 1, len(data_loader), loss.item()))
            writer.add_scalar("Train/iter_loss", loss.cpu().detach().item(), len(data_loader) * epoch + batch_id)
    train_score = np.mean(train_meter.compute_metric(metric))
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, num_epochs, metric, train_score))
    writer.add_scalar(f"Train/{metric}", train_score, epoch)


def predict_logits(model, bg):
    device = get_torch_device()
    bg = bg.to(device)
    # if args['edge_featurizer'] is None:
    # If its a conv graph net or GAT, don't use node info (model.forward() wont take the extra argument)
    if isinstance(model, GCNPredictor) or isinstance(model, GATPredictor):
        node_feats = bg.ndata.pop('h').to(device)
        return model(bg, node_feats)
    elif isinstance(model, GINPredictor):
        node_feats = [
            bg.ndata.pop('atomic_number').to(device),
            bg.ndata.pop('chirality_type').to(device)
        ]
        edge_feats = [
            bg.edata.pop('bond_type').to(device),
            bg.edata.pop('bond_direction_type').to(device)
        ]
        return model(bg, node_feats, edge_feats)
    else:
        node_feats = bg.ndata.pop('h').to(device)
        edge_feats = bg.edata.pop('e').to(device)
        return model(bg, node_feats, edge_feats)


def run_an_eval_epoch(model, data_loader):
    eval_meter = get_prediction_as_meter(model, data_loader)
    rmse = np.mean(eval_meter.compute_metric("rmse"))
    r2 = np.mean(eval_meter.compute_metric("r2"))
    return rmse, r2


def get_prediction_as_meter(model, data_loader):
    device = get_torch_device()
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(device)
            logits = predict_logits(model, bg)
            eval_meter.update(logits, labels, masks)
    return eval_meter


def evaluate_model(model, val_loader):
    val_rmse, val_r2 = run_an_eval_epoch(model, val_loader)
    print('val {} {:.4f}'.format("rmse", val_rmse))
    print('val {} {:.4f}'.format("r2", val_r2))
    return val_rmse, val_r2


def get_predictions(model, loader):
    test_meter = get_prediction_as_meter(model, loader)
    y_pred_batch = torch.cat(test_meter.y_pred, dim=0)
    y_pred = y_pred_batch.reshape(y_pred_batch.shape[0]).tolist()

    return y_pred


def predict(model, smiles, descriptor, batch_size=1, num_workers=general_settings['num_workers']):
    from torch.utils.data import DataLoader
    from MoleculeACE.benchmark.utils import collate_molgraphs
    test_set = get_moleculecsv_dataset(smiles, [0] * len(smiles), descriptor=descriptor)

    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_molgraphs, num_workers=num_workers)

    y_pred = get_predictions(model, test_loader)

    return y_pred
