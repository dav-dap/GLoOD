import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import glood.src.utils.env as env

@hydra.main(version_base=None, config_path="conf", config_name="plot_generic")
def main(cfg: DictConfig) -> None:
    # Persist the resolved Hydra configuration.
    with (env.out() / "conf.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    instantiate(cfg.plotters)

if __name__ == "__main__":
    main()

