from __future__ import annotations

import csv
from pathlib import Path

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch

import glood.src.utils.env as env
from glood import load as load_entrypoint


def hack_conf(cfg: DictConfig) -> DictConfig:
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


def _metric_name(metric: object, idx: int) -> str:
    name = getattr(metric, "__name__", None)
    if name:
        return str(name)
    func = getattr(metric, "func", None)
    if func is not None:
        func_name = getattr(func, "__name__", None)
        if func_name:
            return str(func_name)
    return f"metric_{idx}"


def _example_data_dirs(base_dir: Path) -> list[Path]:
    data_dirs: list[Path] = []
    for example_dir in base_dir.glob("example_*"):
        if not example_dir.is_dir():
            continue
        data_dir = example_dir / "epoch_0" / "data"
        if data_dir.is_dir() and (data_dir / "pred.npy").is_file() and (data_dir / "true.npy").is_file():
            data_dirs.append(data_dir)

    def _sort_key(path: Path) -> int:
        example_name = path.parent.parent.name
        try:
            return int(example_name.removeprefix("example_"))
        except ValueError:
            return 10**12

    return sorted(data_dirs, key=_sort_key)


def _as_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _write_summary_csv(path: Path, values: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(values.keys()))
        writer.writeheader()
        writer.writerow(values)


def _summarize_saved_infer_outputs(
    *,
    infer_base_dir: Path,
    loss_fn: object,
    metrics: tuple[object, ...],
) -> dict[str, float | int]:
    data_dirs = _example_data_dirs(infer_base_dir)
    if not data_dirs:
        raise ValueError(f"No inference outputs found under {infer_base_dir}.")

    loss_total = 0.0
    metric_totals = [0.0 for _ in metrics]

    for data_dir in data_dirs:
        pred = np.load(data_dir / "pred.npy", allow_pickle=True)
        true = np.load(data_dir / "true.npy", allow_pickle=True)
        pred_tensor = torch.from_numpy(pred).unsqueeze(0)
        true_tensor = torch.from_numpy(true).unsqueeze(0)

        loss_total += _as_float(loss_fn(pred_tensor, true_tensor))
        for idx, metric_fn in enumerate(metrics):
            metric_totals[idx] += _as_float(metric_fn(pred_tensor, true_tensor))

    n_examples = len(data_dirs)
    summary: dict[str, float | int] = {
        "n_examples": n_examples,
        "loss": loss_total / n_examples,
    }
    metric_names = tuple(_metric_name(metric_fn, idx) for idx, metric_fn in enumerate(metrics))
    for name, total in zip(metric_names, metric_totals):
        summary[name] = total / n_examples
    return summary


@hydra.main(version_base=None, config_path="conf", config_name="infer_generic")
def main(cfg: DictConfig) -> None:
    # Hack the conf to allow oeperations on variables before instantiating anyting
    cfg = hack_conf(cfg)

    # Persist the resolved Hydra configuration.
    with (env.out() / "conf.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    load_cfg = OmegaConf.create({"input_dir": str(cfg.model_folder)})
    model_wrapper = load_entrypoint.main(load_cfg)
    trained_cfg = OmegaConf.load(env.cwd() / cfg.model_folder / "conf.yaml")
    metrics_cfg = trained_cfg.get("hyperparams_model", {}).get("metrics")
    metrics = tuple(instantiate(metrics_cfg)) if metrics_cfg is not None else ()

    infer_fn = instantiate(cfg.model_wrapper_infer)
    # Hydra instantiates an unbound ModelWrapper.infer partial, so we pass the wrapper
    # instance explicitly to supply the missing self parameter.
    infer_fn(model_wrapper, cfg.plotters)

    infer_base_dir = env.out() / "infer"
    summary = _summarize_saved_infer_outputs(
        infer_base_dir=infer_base_dir,
        loss_fn=model_wrapper.loss_fn,
        metrics=metrics,
    )
    summary_path = infer_base_dir / "metrics_summary.csv"
    _write_summary_csv(summary_path, summary)

    print(f"--> Inference summary saved: {summary_path}")
    print(
        "--> Inference aggregated stats: "
        + ", ".join(
            f"{name}={value}" if name == "n_examples" else f"{name}={value:.10f}"
            for name, value in summary.items()
        )
    )


if __name__ == "__main__":
    main()
