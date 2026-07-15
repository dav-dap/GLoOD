from __future__ import annotations
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

import glood.src.utils.env as env

def hack_conf (cfg: DictConfig) -> DictConfig:
    # Hacking for model: only for glood-like models (viz.: with a backbone)
    model_cfg = cfg.get("model")
    if "backbone" in model_cfg:
        with open_dict(cfg):
            cfg.model.backbone.transformer.d_model = cfg.hyperparams_model.single_latent_dim * 3
            cfg.model.backbone.transformer.dim_feedforward = cfg.hyperparams_model.single_latent_dim * 6
            cfg.model.backbone.sequence_length = cfg.dataset_shapes.len_seq - 1
            cfg.model.n_seq = cfg.dataset_shapes.len_seq - 1

    # Hacking for dataset
    dataset_cfg = OmegaConf.load(
        env.cwd()
        / cfg.hyperparams_dataset.dataset_base_folder
        / ".hydra"
        / "config.yaml"
    )
    with open_dict(cfg):
        cfg.plotters = dataset_cfg.plotters

    return cfg


@hydra.main(version_base=None, config_path="conf", config_name="train_generic")
def main(cfg: DictConfig) -> None:
    # Hack the conf to allow oeperations on variables before instantiating anyting
    cfg = hack_conf(cfg)

    # Persist the resolved Hydra configuration.
    with (env.out() / "conf.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    model_wrapper = instantiate(cfg.model_wrapper)

    train_fn = instantiate(cfg.model_wrapper_train)
    # Hydra instantiates an unbound ModelWrapper.train partial, so we pass the wrapper
    # instance explicitly to supply the missing self parameter.
    train_fn(model_wrapper, cfg.plotters)

    model_wrapper.save_model()

if __name__ == "__main__":
    main()
