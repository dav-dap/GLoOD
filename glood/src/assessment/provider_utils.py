from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import ListConfig, OmegaConf


def _flatten_providers(obj):
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _flatten_providers(item)
        return
    yield obj


def _resolve_provider_item(item):
    if isinstance(item, str):
        providers_cfg_path = Path(__file__).resolve().parents[2] / "conf" / "plotting_providers" / f"{item}.yaml"
        if not providers_cfg_path.is_file():
            raise FileNotFoundError(f"Unknown plotting provider config: {item!r} ({providers_cfg_path}).")
        return instantiate(OmegaConf.load(providers_cfg_path))
    return instantiate(item)


def instantiate_providers(cfg_plotting_providers) -> list:
    if isinstance(cfg_plotting_providers, (list, tuple, ListConfig)):
        instantiated = [_resolve_provider_item(item) for item in cfg_plotting_providers]
        return list(_flatten_providers(instantiated))
    return list(_flatten_providers(_resolve_provider_item(cfg_plotting_providers)))
