from omegaconf import OmegaConf

import app.models as models
from app import create_app

if __name__ == "__main__":
    print("===[Start]===")
    configs_options = [
        "ml_tongji_outcome_kf10.yaml",
        "ml_hm_los_kf10.yaml",
        "gru_tongji_outcome_ep100_kf10_bs64.yaml",
        "gru_tongji_los_ep100_kf10_bs64.yaml",
        "transformer_hm_multitask_ep100_kf10_bs64.yaml.yml",
    ]
    my_pipeline = OmegaConf.load("configs/" + configs_options[1])
    cfg = create_app(my_pipeline)
    print("===[End]===")
