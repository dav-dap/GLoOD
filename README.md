# GLoOD: Generating fLOws frOm Data

GLoOD is a PyTorch training stack for learning data-driven models of unsteady flow fields.  
It combines Hydra for hierarchical configuration, hslur for repeatable job submission, and
standalone entry points for dataset preparation, training, evaluation, and checkpoint
management.

## Highlights
- End-to-end CLI workflow (`glood_build_dataset`, `glood_train`, `glood_plot`, `glood_load`)
  powered by hslur overrides of Hydra configurations.
- Analysis entry points for remote inference data, including drag-force assessment, fixed-section
  profile errors, and square-domain schematics.
- Dataset tooling that converts plain-text CFD snapshots into sharded `.pt` tensors with a
  JSON index for fast discovery.
- Transformer-based sequence model (`glood/src/models/glood.py`) wrapped in a compact CUDA-only
  training harness (`glood/src/managing/model_wrapper.py`).
- Plotting providers and protocols that turn training runs into publication-ready figures.
- Drop-in hslur job descriptors (`config/*.yaml`) for either SLURM clusters or local runs.

## Repository Layout
- `glood/`: Python package with entry points (`build_dataset.py`, `train.py`, etc.) and
  implementation modules under `src/`.
- `glood/conf/`: Hydra configuration tree (model variants, dataset backends, plotting
  pipelines, loss definitions).
- `config/`: Ready-to-use hslur job specs showing how to launch the canonical commands.
- `git_ignore/`: Scratch space populated with toy datasets, saved runs, and cached artifacts.
- `build_torch_128.sh`: Convenience script that creates a CUDA-enabled Conda environment and
  installs PyTorch + project dependencies.
- `install.sh` / `uninstall.sh`: Minimal helpers for editable installs.

## Requirements
- Python 3.10+ with an NVIDIA GPU (training enforces `torch.cuda.is_available()`).
- PyTorch (CUDA build), Hydra Core, OmegaConf, and hslur.
- Optional: Conda/Mamba for using `build_torch_128.sh`.

### Quick install
```bash
bash build_torch_128.sh
source activate env-torch-128
pip install -e .
```

## Command Line Entry Points
All executables honor Hydra syntax for overrides and compose their configuration from
`glood/conf/`.

- `glood_build_dataset`: Parse raw TXT snapshots into sharded PyTorch tensors and create a
  diagnostic plot bundle.
- `glood_train`: Instantiate the configured model, train it on sharded data, and save weights,
  optimizer, scheduler, and resolved configuration.
- `glood_process_stats`: Aggregate multiple `history.csv` logs (e.g., k-fold runs) into mean/std
  tables and plot loss/metric curves with error bands (PNG/PDF).
- `glood_plot`: Re-run plotting protocols on a completed run (training or evaluation).
- `glood_load`: Materialize a `ModelWrapper` and load weights from an existing run directory.
- `glood_assess_profiles`: Compute fixed-section `u_x(y)` profile errors on inference datasets,
  export CSV summaries, and plot selected line profiles.
- `glood_assess_fieldwise`: Run a whole-field diagnostic that decomposes `ux`, `uy`, and
  `speed` errors into raw, scale-corrected, shift-corrected, and shift+scale-corrected views.
- `glood_assess_problematic_examples`: Audit one inference dataset for suspicious snapshots with
  unusually high gradients or oscillations, or audit sharded train/test datasets through the same
  entry point by swapping the Hydra `problematic_input_source` config group.
- `glood_export_fieldwise_examples`: Extract only a small selected set of whole-field PNG panels
  from remote inference tarballs, preserving dataset/example/timestep identifiers in filenames.
- `glood_assess_drag_force_remote`: Re-run drag-force analysis on a configured list of remote
  inference folders and write one labeled output bundle per dataset.
- `glood_plot_square_domain_sections`: Draw a square computational-domain schematic with the
  configured measurement sections.

Each command writes artifacts to `RUN_DIR` (defaults to `.`). The project root defaults to
`SUBMIT_DIR` (also `.` without overrides); hslur populates both when launching jobs.

## hslur Configuration Catalog
The `config/` folder contains ready-made hslur job descriptors that exercise the full command
suite. Launch them with `hslur config=<config-path>`.

- `config/test_build_dataset_seq.yaml`: Builds multi-frame shards from TXT snapshots using
  `parsing=series` and `plotters=series`. Targets the demonstration inputs in
  `git_ignore/test_dataset/simulations`, writes shards to the configured sandbox, and emits
  diagnostics for sequential data.
- `config/test_build_dataset_single.yaml`: Uses `parsing=single` with the matching plotting
  stack to generate L=1 shards. Shows how to stretch `diagnostics.examples_to_plot` when you
  want dense visual checks for steady-state captures.
- `config/test_train_glood.yaml`: Trains the transformer-backed model on sequential shards,
  wiring up `hyperparams_dataset.shard_dir`, optimizer hyperparameters, and a validation split
  (`val_fraction=0.8`). Serves as the canonical end-to-end training recipe.
- `config/test_train_unet_tri_decoupled.yaml`: Swaps in the `unet_tri_decoupled` backbone and
  single-frame dataset to stress-test the visual model independently of the transformer.
- `config/test_plot.yaml`: Replays plotting protocols against artifacts in `git_ignore/test_eval`
  with providers pulled from `provider_file_generic` and presentation tweaks such as
  `plotting.remove_title=true`.
- `config/test_load.yaml`: Restores a saved checkpoint from `git_ignore/test_saved_model_seq`
  without retraining, verifying that model and optimizer state can be reloaded in isolation.
- `config/assess_profiles_remote_matnum.yaml`: Launches the profile-error analysis on the remote
  `swin/unet x cavity/circles/squares` inference datasets listed in
  `glood/conf/assessment_inputs/remote_matnum_all.yaml`, accepting either extracted directories
  or tar archives with the standard `example_*/epoch_*/data/*.npy` layout.
- `config/assess_fieldwise_remote_matnum.yaml`: Launches the whole-field diagnostic on the same
  remote inference tarballs across `cavity/circles/squares`.
- `config/assess_problematic_examples_remote_matnum.yaml`: Audits the remote cavity matnum shards
  through `problematic_input_source=sharded_dataset`, using the same dataset pipeline that feeds
  train/infer jobs.
- `config/assess_problematic_examples_inference_cavity_matnum.yaml`: Audits the cavity fields
  stored in inference tarballs through `problematic_input_source=inference_npy`; inference labels
  are inferred from `input_dir`.
- `config/export_fieldwise_examples_remote_matnum.yaml`: Extracts the selected whole-field panels
  referenced by the fieldwise selected-example manifest, without unpacking the full tarballs.
- `config/assess_drag_force_remote_matnum.yaml`: Launches the remote drag-force batch analysis,
  preserving the historical CSV layout while reading the remote inference tarballs from the same
  shared dataset config.
- `config/plot_square_domain_sections.yaml`: Emits the square-domain schematic and the companion
  CSV containing the section locations.

## Configuration Notes
- Hydra defaults live under `glood/conf/`. Every placeholder marked `__DUMMY__` must be provided
  either via hslur / CLI overrides or via a custom YAML in `conf/`.
- Common overrides:
  - `hyperparams_dataset.shard_dir`: Directory (relative to `SUBMIT_DIR`) containing `.pt`
    shards.
  - `hyperparams_model.single_latent_dim`: Size of the latent space used by the U-Net encoder
    and transformer backbone (also governs transformer dimensions via `hack_conf`).
  - `hyperparams_model.learning_rate`, `hyperparams_model.dropout_rate`, `train_params.*`,
    `scheduler.kwargs.*`.
- Custom compose files can be dropped in any subfolder of `glood/conf/` following Hydra naming
  conventions. For example, creating `glood/conf/train/airfoil.yaml` enables
  `glood_train +train=airfoil`.
- Remote analysis datasets are listed explicitly in
  `glood/conf/assessment_inputs/remote_matnum_all.yaml`; both drag-force reruns and profile
  analysis accept either extracted inference directories or tar archives containing the same
  `example_*/epoch_*/data/*.npy` structure.
- Use `hydra.run.dir=.` to force in-place execution or override it with an absolute path if you
  prefer Hydra-managed output directories.

## Working with hslur
hslur wraps Hydra jobs with launch instructions (local or SLURM). The sample descriptors in
`config/` illustrate the pattern:

```bash
# Dry-run locally with the provided toy dataset
hslur config=config/test_train_glood.yaml

# Submit to a SLURM queue after editing the slurm: block
hslur config=config/test_train_glood.yaml local=false
```

Each descriptor specifies:
- `cmd`: Which project entry point to execute (e.g., `glood_train`).
- `prelude`: Environment setup commands (module loads, `conda activate`, etc.).
- `sandbox_root`: Where run directories are staged when sandboxing is enabled.
- `program`: Hydra overrides applied to the entry point.

When hslur launches a job it sets `SUBMIT_DIR` and `RUN_DIR` automatically, so artifacts end up
inside the sandbox without additional flags.

## Sample Assets
- `git_ignore/test_dataset/`: Small synthetic dataset used by the example commands.
- `git_ignore/test_dataset_seq/`: Minimal sharded dataset for smoke tests.
- `config/test_*.yaml`: End-to-end examples for dataset building, training, plotting, and
  checkpoint loading.

## Troubleshooting
- **Hydra complains about `__DUMMY__`:** Provide the missing override on the CLI or in a custom
  config file.
- **`ValueError: ModelWrapper only supports CUDA devices`:** Activate a CUDA-enabled PyTorch
  build and make sure `torch.cuda.is_available()` returns `True`.
- **No `.pt` shards produced:** Check that `input_dir` points to directories containing TXT
  files with the expected number of floats per line (`H * W * C`).
- **Plots look empty:** Confirm that the plotting providers have access to the `train/` and
  `val/` subdirectories generated during training, or re-run plotting with the correct
  `input_dir`.
