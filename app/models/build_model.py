import torch
from torch import nn

from .backbones import *
from .classifiers import *
from .losses import *


class Model:
    def __init__(self, backbone, head):
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def build_backbone_from_cfg(cfg):
    if cfg.model == "transformer":
        return Transformer(
            lab_dim=cfg.lab_dim,
            demo_dim=cfg.demo_dim,
            max_visits=cfg.max_visits,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
        )


def build_classifier_from_cfg(cfg):
    if cfg.task == "los":
        return LosHead(
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            act_layer=cfg.act_layer,
            drop=cfg.drop,
        )
    elif cfg.task == "outcome":
        return OutcomeHead(
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            act_layer=cfg.act_layer,
            drop=cfg.drop,
        )
    else:
        raise ValueError("Unknown task: {}".format(cfg.task))


def build_model_from_cfg(cfg):
    backbone = build_backbone_from_cfg(cfg)
    head = build_classifier_from_cfg(cfg)
    return Model(backbone, head)
