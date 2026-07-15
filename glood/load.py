from __future__ import annotations

import shutil
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import glood.src.utils.env as env


@hydra.main(version_base=None, config_path="conf", config_name="load_generic")
def main(cfg: DictConfig) -> None:
    # Persist the resolved Hydra configuration.
    with (env.out() / "conf_load.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    input_dir = env.cwd() / cfg.input_dir

    accessory_cfg_path = input_dir / "conf.yaml"
    if not accessory_cfg_path.is_file():
        raise FileNotFoundError(f"Expected configuration file at {accessory_cfg_path}.")

    shutil.copy2(accessory_cfg_path, env.out() / "conf.yaml")

    accessory_cfg = OmegaConf.load(accessory_cfg_path)

    model_wrapper = instantiate(accessory_cfg.model_wrapper)
    model_wrapper.load_model(input_dir)

    return model_wrapper


if __name__ == "__main__":
    main()
