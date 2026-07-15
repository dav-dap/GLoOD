.PHONY: all \
	clean \

TR = git_ignore/test_runs

# TARGETS

all: \
	dataset_single \
	dataset_seq \
	dataset_seq_matnum \
	train_unet \
	train_swin \
	train_glood_unet \
	train_identity_unet \
	train_glood_unet_matnum \
	train_glood_unet_kfold \
	load_unet \
	load_glood_unet \
	plot_single \
	plot_seq \
	plot_seq_matnum \
	infer_unet \
	infer_glood_unet \
	infer_glood_unet_matnum \
	assess_boundaries \
	plot_assess_boundaries_sweep \
	assess_drag_force \
	process_stats

dataset_single: $(TR)/dataset_single
dataset_seq: $(TR)/dataset_seq
dataset_seq_matnum: $(TR)/dataset_seq_matnum
train_unet: $(TR)/train_unet
train_swin: $(TR)/train_swin
train_glood_unet: $(TR)/train_glood_unet
train_identity_unet: $(TR)/train_identity_unet
train_glood_unet_matnum: $(TR)/train_glood_unet_matnum
train_glood_unet_kfold: $(TR)/train_glood_unet_kfold
load_unet: $(TR)/load_unet
load_glood_unet: $(TR)/load_glood_unet
plot_single: $(TR)/plot_single
plot_seq: $(TR)/plot_seq
plot_seq_matnum: $(TR)/plot_seq_matnum
infer_unet: $(TR)/infer_unet
infer_glood_unet: $(TR)/infer_glood_unet
infer_glood_unet_matnum: $(TR)/infer_glood_unet_matnum
assess_boundaries: $(TR)/assess_boundaries
plot_assess_boundaries_sweep: $(TR)/plot_assess_boundaries_sweep
assess_drag_force: $(TR)/assess_drag_force
process_stats: $(TR)/process_stats

# IMPLEMENTATIONS

$(TR)/dataset_single:
	@echo "============================================================"
	hslur config=test_config/dataset_single.yaml

$(TR)/dataset_seq:
	@echo "============================================================"
	hslur config=test_config/dataset_seq.yaml

$(TR)/dataset_seq_matnum:
	@echo "============================================================"
	hslur config=test_config/dataset_seq.yaml \
		sandbox_root=$(TR)/dataset_seq_matnum \
		parsing=series_with_matnum \
		dataset.shapes.C=4 \
		input_dir=test_material/simulations_matnum

$(TR)/train_unet: | dataset_single
	@echo "============================================================"
	hslur config=test_config/train_unet.yaml

$(TR)/train_swin: | dataset_single
	@echo "============================================================"
	hslur config=test_config/train_swin.yaml

$(TR)/train_glood_unet: | dataset_seq
	@echo "============================================================"
	hslur config=test_config/train_glood_unet.yaml

$(TR)/train_identity_unet: | dataset_seq
	@echo "============================================================"
	hslur config=test_config/train_identity_unet.yaml

$(TR)/train_glood_unet_matnum: | dataset_seq_matnum
	@echo "============================================================"
	hslur config=test_config/train_glood_unet.yaml \
		sandbox_root=$(TR)/train_glood_unet_matnum \
		hyperparams_dataset.dataset_base_folder=$(TR)/dataset_seq_matnum/0 \
		dataset=sharded_series_with_matnum

$(TR)/train_glood_unet_kfold: | dataset_seq
	@echo "============================================================"
	hslur config=test_config/train_glood_unet_kfold.yaml

$(TR)/load_unet: | train_unet
	@echo "============================================================"
	hslur config=test_config/load.yaml \
		sandbox_root=$(TR)/load_unet input_dir=$(TR)/train_unet/0

$(TR)/load_glood_unet: | train_glood_unet
	@echo "============================================================"
	hslur config=test_config/load.yaml

$(TR)/plot_single: | train_unet
	@echo "============================================================"
	hslur config=test_config/plot.yaml \
		sandbox_root=$(TR)/plot_single plotters=single \
		input_dir=$(TR)/train_unet/0/train

$(TR)/plot_seq: | train_glood_unet
	@echo "============================================================"
	hslur config=test_config/plot.yaml

$(TR)/plot_seq_matnum: | train_glood_unet_matnum
	@echo "============================================================"
	hslur config=test_config/plot_matnum.yaml \
		sandbox_root=$(TR)/plot_seq_matnum \
		input_dir=$(TR)/train_glood_unet_matnum/0/train

$(TR)/infer_unet: train_unet
	@echo "============================================================"
	hslur config=test_config/infer_glood_unet.yaml \
		sandbox_root=$(TR)/infer_unet \
		hyperparams_dataset.dataset_base_folder=$(TR)/dataset_single/0 \
		model_folder=git_ignore/test_runs/train_unet/0 \
		dataset=sharded_single

$(TR)/infer_glood_unet: train_glood_unet
	@echo "============================================================"
	hslur config=test_config/infer_glood_unet.yaml

$(TR)/infer_glood_unet_matnum: train_glood_unet_matnum
	@echo "============================================================"
	hslur config=test_config/infer_glood_unet.yaml \
		sandbox_root=$(TR)/infer_glood_unet_matnum \
		hyperparams_dataset.dataset_base_folder=$(TR)/dataset_seq_matnum/0 \
		model_folder=git_ignore/test_runs/train_glood_unet_matnum/0 \
		dataset=sharded_series_with_matnum

$(TR)/assess_boundaries: | train_glood_unet_matnum
	@echo "============================================================"
	hslur config=test_config/assess_boundaries.yaml \
		sandbox_root=$(TR)/assess_boundaries \
		input_dir=$(TR)/train_glood_unet_matnum/0/train \
		assessment.epsilon=1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12

$(TR)/plot_assess_boundaries_sweep: | assess_boundaries
	@echo "============================================================"
	hslur config=test_config/plot_assess_boundaries_sweep.yaml \
		sandbox_root=$(TR)/plot_assess_boundaries_sweep \
		input_dir=$(TR)/assess_boundaries

$(TR)/assess_drag_force: | train_glood_unet_matnum
	@echo "============================================================"
	hslur config=test_config/assess_drag_force.yaml \
		sandbox_root=$(TR)/assess_drag_force \
		input_dir=$(TR)/train_glood_unet_matnum/0/train

$(TR)/process_stats: | train_glood_unet_kfold
	@echo "============================================================"
	hslur config=test_config/process_stats.yaml

clean:
	rm -fr $(TR)
