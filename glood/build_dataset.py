from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import glood.src.utils.env as env


def _discover_snapshot_txt_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("**/snapshots/**/*.txt") if path.is_file())


@hydra.main(version_base=None, config_path="conf", config_name="build_dataset_generic")
def main(cfg: DictConfig) -> None:
    # Persist the resolved Hydra configuration.
    with (env.out() / "conf.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    txt_files = _discover_snapshot_txt_files(env.cwd() / cfg.input_dir)
    
    if not txt_files:
        raise ValueError(f"No .txt files found in {cfg.input_dir}")
    print(f"Found {len(txt_files)} snapshot text files in {cfg.input_dir}")
    
    parser = instantiate(cfg.parsing)

    # Converting the series of txt files into .pt dataset shards
    parser.txt2pt(txt_files, cfg.dataset.examples_per_file)
    
    # Plotting diagnostics
    instantiate(cfg.plotters)

if __name__ == "__main__":
    main()
