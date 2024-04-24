import math
import pathlib
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    SubsetRandomSampler,
    TensorDataset,
    random_split,
)

from app.core.evaluation import covid_metrics, eval_metrics
from app.core.utils import init_random
from app.datasets import get_dataset, load_data, load_data_split
from app.datasets.dl import Dataset
from app.datasets.ml import flatten_dataset, numpy_dataset
from app.models import (
    build_model_from_cfg,
    get_multi_task_loss,
    predict_all_visits_bce_loss,
    predict_all_visits_mse_loss,
)
from app.utils import perflog
from tqdm import tqdm
import wandb


def train_epoch(model, device, dataloader, loss_fn, optimizer, info):
    train_loss = []
    model.train()
    for step, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Train", leave=True):
        batch_x, batch_y, batch_x_lab_length = data
        batch_x, batch_y, batch_x_lab_length = (
            batch_x.float().to(device),
            batch_y.float().to(device),
            batch_x_lab_length.float().to(device),
        )
        # batch_y = batch_y[:, :]  # 0: outcome, 1: los
        batch_y = batch_y[:, :, 0]  # 0: outcome, 1: los
        batch_y = batch_y.unsqueeze(-1)
        optimizer.zero_grad()
        output = model(batch_x, device, info)
        output = output[:, -1, :].unsqueeze(1)
        loss = loss_fn(output, batch_y, batch_x_lab_length)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.array(train_loss).mean()


def val_epoch(model, device, dataloader, loss_fn, info):
    """
    val / test
    """
    val_loss = []
    y_pred = []
    y_true = []
    y_true_all = []
    len_list = []
    model.eval()
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            batch_x, batch_y, batch_x_lab_length = data
            batch_x, batch_y, batch_x_lab_length = (
                batch_x.float().to(device),
                batch_y.float().to(device),
                batch_x_lab_length.float().to(device),
            )
            all_y = batch_y
            batch_y = batch_y[:, :, 0]  # 0: outcome, 1: los
            batch_y = batch_y.unsqueeze(-1)
            output = model(batch_x, device, info)
            output = output[:, -1, :].unsqueeze(1)
            loss = loss_fn(output, batch_y, batch_x_lab_length)
            val_loss.append(loss.item())
            len_list.extend(batch_x_lab_length.long().tolist())
            for i in range(len(batch_y)):
                y_pred.extend(
                    output[i][: batch_x_lab_length[i].long()].tolist())
                y_true.extend(
                    batch_y[i][: batch_x_lab_length[i].long()].tolist())
                y_true_all.extend(
                    all_y[i][: batch_x_lab_length[i].long()].tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_all = np.array(y_true_all)
    len_list = np.array(len_list)
    # print("len:", len(y_true), len_list.sum(), len_list)
    # early_prediction_score = covid_metrics.early_prediction_outcome_metric(
    #     y_true_all,
    #     y_pred,
    #     len_list,
    #     info["config"].thresholds,
    #     verbose=0,
    # )
    y_pred = np.stack([1 - y_pred, y_pred], axis=1)
    evaluation_scores = eval_metrics.print_metrics_binary(
        y_true, y_pred, verbose=0)
    # evaluation_scores["early_prediction_score"] = early_prediction_score
    return np.array(val_loss).mean(), evaluation_scores


def start_pipeline(cfg, device):
    print(f'start pipeline, outcome split')
    info = {"config": cfg, "epoch": 0}
    val_info = {"config": cfg, "epoch": cfg.epochs}
    dataset_type, method, num_folds, train_fold = (
        cfg.dataset,
        cfg.model,
        cfg.num_folds,
        cfg.train_fold,
    )
    # Load data
    train_xs, train_ys, train_lens, valid_xs, valid_ys, valid_lens, test_xs, test_ys, test_lens = load_data_split(
        dataset_type=dataset_type)
    train_dataset = get_dataset(train_xs, train_ys, train_lens)
    valid_dataset = get_dataset(valid_xs, valid_ys, valid_lens)
    test_dataset = get_dataset(test_xs, test_ys, test_lens)

    all_history = {}
    test_performance = {
        "test_loss": [],
        "test_accuracy": [],
        "test_auroc": [],
        "test_auprc": [],
        "test_minpse": [],
        "test_f1_score": [],
        "test_prec0": [],
        "test_prec1": [],
        "test_rec0": [],
        "test_rec1": [],
    }

    # train
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
    )
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_auroc": [],
        "val_auprc": [],
        "val_minpse": [],
        "val_f1_score": [],
        "val_prec0": [],
        "val_prec1": [],
        "val_rec0": [],
        "val_rec1": [],

    }
    print(cfg.model_init_seed)
    print(cfg.train)
    if cfg.wandb:
        wandb.init(project=cfg.wandb_project, config=dict(cfg))
        # wandb.config.update(cfg)
    for seed in cfg.model_init_seed:
        init_random(seed)
        model = build_model_from_cfg(cfg, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = predict_all_visits_bce_loss
        best_val_performance = 0.0

        if cfg.train == True:
            for epoch in range(cfg.epochs):
                info["epoch"] = epoch + 1
                train_loss = train_epoch(
                    model,
                    device,
                    train_loader,
                    criterion,
                    optimizer,
                    info=info,
                )
                val_loss, val_evaluation_scores = val_epoch(
                    model,
                    device,
                    val_loader,
                    criterion,
                    info=val_info,
                )
                # save performance history on validation set
                print(
                    "Epoch:{}/{} AVG Training Loss:{:.4f} AVG Val Loss:{:.4f}".format(
                        epoch + 1, cfg.epochs, train_loss, val_loss
                    )
                )
                if cfg.wandb:
                    wandb.log(
                        {
                            "train/loss": train_loss,
                            "val/loss": val_loss, })
                    for key, value in val_evaluation_scores.items():
                        wandb.log({f"val/{key}": value}, commit=False)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_evaluation_scores["acc"])
                history["val_auroc"].append(val_evaluation_scores["auroc"])
                history["val_auprc"].append(val_evaluation_scores["auprc"])
                history["val_minpse"].append(val_evaluation_scores["minpse"])
                history["val_f1_score"].append(
                    val_evaluation_scores["f1_score"])
                history["val_prec0"].append(val_evaluation_scores["prec0"])
                history["val_prec1"].append(val_evaluation_scores["prec1"])
                history["val_rec0"].append(val_evaluation_scores["rec0"])
                history["val_rec1"].append(val_evaluation_scores["rec1"])
                # if auroc is better, than set the best auroc, save the model, and test it on the test set
                if val_evaluation_scores["auprc"] > best_val_performance:
                    best_val_performance = val_evaluation_scores["auprc"]
                    torch.save(
                        model.state_dict(),
                        f"checkpoints/{cfg.name}_seed{seed}.pth",
                    )
                    print("[best!!]", epoch,
                          f'best val_auprc: {best_val_performance}')
                    es = 0
                else:
                    es += 1
                    if es >= 5:
                        print(f"Early stopping break at epoch {epoch}")
                        break

        print(f"Load best model for seed {seed}")
        model = build_model_from_cfg(cfg, device)
        model.load_state_dict(
            torch.load(
                f"checkpoints/{cfg.name}_seed{seed}.pth",
                map_location=torch.device("cpu"),
            )
        )
        test_loss, test_evaluation_scores = val_epoch(
            model,
            device,
            test_loader,
            criterion,
            info=val_info,
        )
        print(f'test result {test_evaluation_scores}')
        test_performance["test_loss"].append(test_loss)
        test_performance["test_accuracy"].append(test_evaluation_scores["acc"])
        test_performance["test_auroc"].append(test_evaluation_scores["auroc"])
        test_performance["test_auprc"].append(test_evaluation_scores["auprc"])
        test_performance["test_minpse"].append(
            test_evaluation_scores["minpse"])
        test_performance["test_f1_score"].append(
            test_evaluation_scores["f1_score"])
        test_performance["test_prec0"].append(test_evaluation_scores["prec0"])
        test_performance["test_prec1"].append(test_evaluation_scores["prec1"])
        test_performance["test_rec0"].append(test_evaluation_scores["rec0"])
        test_performance["test_rec1"].append(test_evaluation_scores["rec1"])
        print(
            f"Test Loss = {test_loss}, \
            ACC = {test_evaluation_scores['acc']}, \
            AUROC = {test_evaluation_scores['auroc']}, \
            AUPRC = {test_evaluation_scores['auprc']}"
        )
        if cfg.wandb:
            wandb.log({"test/loss": test_loss, }, commit=False)
            for key, value in test_evaluation_scores.items():
                wandb.log({f"test/{key}": value}, commit=False)
    # Calculate average performance on 10-fold test set
    test_accuracy_list = np.array(test_performance["test_accuracy"])
    test_auroc_list = np.array(test_performance["test_auroc"])
    test_auprc_list = np.array(test_performance["test_auprc"])

    print("====================== TEST RESULT ======================")
    print(
        "ACC: {:.4f} ({:.4f})".format(
            test_accuracy_list.mean(), test_accuracy_list.std()
        )
    )
    print(
        "AUROC: {:.4f} ({:.4f})".format(
            test_auroc_list.mean(), test_auroc_list.std())
    )
    print(
        "AUPRC: {:.4f} ({:.4f})".format(
            test_auprc_list.mean(), test_auprc_list.std())
    )
    print("minpse: {:.4f} ({:.4f})".format(float(np.mean(
        test_performance["test_minpse"])), float(np.std(test_performance["test_minpse"]))))
    print("f1_score: {:.4f} ({:.4f})".format(float(np.mean(
        test_performance["test_f1_score"])), float(np.std(test_performance["test_f1_score"]))))
    print("prec0: {:.4f} ({:.4f})".format(float(np.mean(
        test_performance["test_prec0"])), float(np.std(test_performance["test_prec0"]))))
    print("prec1: {:.4f} ({:.4f})".format(float(np.mean(
        test_performance["test_prec1"])), float(np.std(test_performance["test_prec1"]))))
    print("rec0: {:.4f} ({:.4f})".format(float(np.mean(
        test_performance["test_rec0"])), float(np.std(test_performance["test_rec0"]))))
    print("rec1: {:.4f} ({:.4f})".format(float(np.mean(
        test_performance["test_rec1"])), float(np.std(test_performance["test_rec1"]))))
    print("=========================================================")
    # perflog.process_and_upload_performance(
    #     cfg,
    #     acc=test_accuracy_list,
    #     auroc=test_auroc_list,
    #     auprc=test_auprc_list,
    #     verbose=1,
    #     upload=cfg.db,
    # )
