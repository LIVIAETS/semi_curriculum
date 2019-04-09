CC = python3.7
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report
.PRECIOUS: results/learn_size/pred_size_%.pkl data/ACDC-2D-LS/%

# CFLAGS = -O
# DEBUG = --debug
EPC = 100
# EPC = 5

G_RGX = (patient\d+_\d+_\d+)_\d+
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]
NET = ENet
KEPT = 25

TRN = results/learn_size/fs_75 results/learn_size/fs_40 results/learn_size/fs_30 \
			results/learn_size/fs_20 results/learn_size/fs_10 results/learn_size/fs_5 \
		results/learn_size/semi_oracle_size_40 results/learn_size/semi_oracle_size_30 \
			results/learn_size/semi_oracle_size_20 results/learn_size/semi_oracle_size_10 results/learn_size/semi_oracle_size_5 \
		results/learn_size/semi_pred_size_40 results/learn_size/semi_pred_size_30 \
			results/learn_size/semi_pred_size_20 results/learn_size/semi_pred_size_10 results/learn_size/semi_pred_size_5 \
		results/learn_size/proposals_40 results/learn_size/proposals_30 \
			results/learn_size/proposals_20 results/learn_size/proposals_10 results/learn_size/proposals_5 \

GRAPH =
# GRAPH = results/learn_size/val_dice.png results/learn_size/tra_dice.png \
# 		results/learn_size/tra_loss.png \
# 		results/learn_size/val_batch_dice.png
# HIST =  results/learn_size/val_dice_hist.png results/learn_size/tra_loss_hist.png \
# 		results/learn_size/val_batch_dice_hist.png
BOXPLOT =
# BOXPLOT = results/learn_size/val_batch_dice_boxplot.png
PLT = $(GRAPH) $(HIST) $(BOXPLOT) results/learn_size/abla_patient_best.png results/learn_size/abla_patient_avg.png

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-learn_size.tar.gz

all: pack

plot: $(PLT)

pack: $(PACK) report
$(PACK): $(PLT) $(TRN)
	mkdir -p $(@D)
	tar cf - $^ results/learn_size/*.pkl | pigz > $@
	chmod -w $@
# tar -zc -f $@ $^  # Use if pigz is not available


data/acdc: data/acdc.lineage data/acdc.zip
	md5sum -c $<
	rm -rf $@_tmp $@
	unzip -q $(word 2, $^) -d $@_tmp
	rm $@_tmp/training/*/*_4d.nii.gz  # space optimization
	mv $@_tmp $@

data/spatial3D_York: data/york.lineage data/spatial3D_York.tar.gz
	sha256sum -c $<
	rm -rf $@_tmp $@
	mkdir -p $@_tmp
	tar xf $(word 2, $^) --directory $@_tmp --strip 1
	mv $@_tmp $@

data/YORK: OPT = --seed=0 --retain 8
data/YORK: data/spatial3D_York
	rm -rf $@_tmp $@
	$(PP) $(CC) $(CFLAGS) preprocess/slice_york.py --source_dir=$< --dest_dir=$@_tmp $(OPT)
	mv $@_tmp $@

# Data generation
## Create the initial train/val split
data/ACDC-2D-LS/val/gt data/ACDC-2D-LS/train/gt: OPT = --seed=1 --retain $(KEPT)
data/ACDC-2D-LS/val/gt data/ACDC-2D-LS/train/gt: DTS = data/ACDC-2D-LS
data/ACDC-2D-LS/val/gt data/ACDC-2D-LS/train/gt: data/acdc
	rm -rf $(DTS)_tmp $(DTS)
	$(PP) $(CC) $(CFLAGS) preprocess/slice_acdc.py --source_dir="data/acdc/training" --dest_dir=$(DTS)_tmp $(OPT)
	$(CC) remap_values.py $(DTS)_tmp/train/gt "{3: 3, 2: 0, 1: 0, 0: 0}"
	$(CC) remap_values.py $(DTS)_tmp/val/gt "{3: 3, 2: 0, 1: 0, 0: 0}"
	mv $(DTS)_tmp $(DTS)

## Sub training splits: _b stands for base
data/ACDC-2D-LS/train_b_%: N=$(subst train_b_,,$(@F))
data/ACDC-2D-LS/train_b_%: S=$(shell echo $$(( $(N) + 1 )))
data/ACDC-2D-LS/train_b_%: data/ACDC-2D-LS/train/gt # Delete at line N+1 onwards
	cp -r $(<D) $@_tmp
	for f in `ls $@_tmp` ; do \
		echo $$f ; \
		for im in `ls $@_tmp/img | cut -d '_' -f 1 | sort | uniq | tail -n +$(S)` ; do \
			rm $@_tmp/$$f/$$im* ; \
		done \
	done
	mv $@_tmp $@
# _r stands for reverse
data/ACDC-2D-LS/train_r_%: N=$(subst train_r_,,$(@F))
data/ACDC-2D-LS/train_r_%: data/ACDC-2D-LS/train/gt # Delete N firsts
	cp -r $(<D) $@_tmp
	for f in `ls $@_tmp` ; do \
		echo $$f ; \
		for im in `ls $@_tmp/img | cut -d '_' -f 1 | sort | uniq | head -n $(N)` ; do \
			rm $@_tmp/$$f/$$im* ; \
		done \
	done
	mv $@_tmp $@

data/ACDC-2D-LS/train_a_%: data/ACDC-2D-LS/train_b_%
	rm -rf $@ $@_tmp
	$(CC) $(CFLAGS) augment.py --n_aug 10 --root_dir $< --dest_dir $@_tmp
	mv $@_tmp $@


## Sub validation splits
data/ACDC-2D-LS/val_%: N=$(subst val_,,$(@F))
data/ACDC-2D-LS/val_%: S=$(shell echo $$(( $(N) + 1 )))
data/ACDC-2D-LS/val_%: data/ACDC-2D-LS/val/gt # Delete at line N+1 onwards
	cp -r $(<D) $@_tmp
	for f in `ls $@_tmp` ; do \
		echo $$f ; \
		for im in `ls $@_tmp/img | cut -d '_' -f 1 | sort | uniq | tail -n +$(S)` ; do \
			rm $@_tmp/$$f/$$im* ; \
		done \
	done
	mv $@_tmp $@
# _r stands for reverse
data/ACDC-2D-LS/val_r_%: N=$(subst val_r_,,$(@F))
data/ACDC-2D-LS/val_r_%: data/ACDC-2D-LS/val/gt # Delete N firsts
	cp -r $(<D) $@_tmp
	for f in `ls $@_tmp` ; do \
		echo $$f ; \
		for im in `ls $@_tmp/img | cut -d '_' -f 1 | sort | uniq | head -n $(N)` ; do \
			rm $@_tmp/$$f/$$im* ; \
		done \
	done
	mv $@_tmp $@


# FS baselines
results/learn_size/fs_%: N=$(subst fs_,,$(@F))
results/learn_size/fs_%: OPT = --losses="[('CrossEntropy', {'idc': [0, 3]}, None, None, None, 1)]" \
		--training_folders train_b_$(N)
results/learn_size/fs_%: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"
results/learn_size/fs_%: data/ACDC-2D-LS/val_20 data/ACDC-2D-LS/train_b_%
	rm -rf $@_tmp
	$(CC) $(CFLAGS) main.py --dataset=$(<D) --batch_size=1 --group --schedule \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=4 --metric_axis 3 \
		--grp_regex="$(G_RGX)" --validation_folder val_20 --in_memory --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@


# Oracle baseline
results/learn_size/semi_oracle_size_%: N=$(subst semi_oracle_size_,,$(@F))
results/learn_size/semi_oracle_size_%: OPT = --losses="[[('CrossEntropy', {'idc': [0, 3]}, None, None, None, 1)], \
			[('NaivePenalty', {'idc': [3]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2)]]" \
		--training_folders train_b_$(N) train_r_$(N) --val_loader_id 0
results/learn_size/semi_oracle_size_%: DATA = --folders="[$(B_DATA)+[('gt', gt_transform, True)], $(B_DATA)+[('gt', dummy_gt, False)]]"
results/learn_size/semi_oracle_size_%: data/ACDC-2D-LS/val_20 data/ACDC-2D-LS/train_b_% data/ACDC-2D-LS/train_r_%
	rm -rf $@_tmp
	$(CC) $(CFLAGS) main.py --dataset=$(<D) --batch_size=1 --group --schedule \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=4 --metric_axis 3 \
		--grp_regex="$(G_RGX)" --validation_folder val_20 --in_memory --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@


# Learn size like a real man
results/learn_size/pred_size_%.pkl: N=$(basename $(subst pred_size_,,$(@F)))
results/learn_size/pred_size_%.pkl: data/ACDC-2D-LS/val_r_20 data/ACDC-2D-LS/train_a_%
	rm -f $@_tmp $@
	$(CC) $(CFLAGS) train_regression.py --n_class 4 --base-width 4 --cardinality 32 --save_dest $@_tmp --data_root $(<D) --epc 200 \
		--train_subfolder train_a_$(N) --val_subfolder val_r_20 --idc 3 --batch_size 10 --in_memory $(DEBUG)
	mv $@_tmp $@

results/learn_size/semi_pred_size_%: N=$(subst semi_pred_size_,,$(@F))
results/learn_size/semi_pred_size_%: OPT = --losses="[[('CrossEntropy', {'idc': [0, 3]}, None, None, None, 1)], \
			[('NaivePenalty', {'idc': [3]}, 'PredictionBounds', {'margin': 0.10, 'mode': 'percentage', 'net': 'results/learn_size/pred_size_$(N).pkl'}, 'soft_size', 1e-2)]]" \
		--training_folders train_b_$(N) train_r_$(N) --val_loader_id 0
results/learn_size/semi_pred_size_%: DATA = --folders="[$(B_DATA)+[('gt', gt_transform, True)], $(B_DATA)+[('gt', dummy_gt, False)]]"
results/learn_size/semi_pred_size_%: data/ACDC-2D-LS/val_20 data/ACDC-2D-LS/train_b_% data/ACDC-2D-LS/train_r_% results/learn_size/pred_size_%.pkl
	rm -rf $@_tmp
	$(CC) $(CFLAGS) main.py --dataset=$(<D) --batch_size=1 --group --schedule \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=4 --metric_axis 3 \
		--grp_regex="$(G_RGX)" --validation_folder val_20 --in_memory --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@


# Proposals
# Train on base (b), predict on reverse (r): p_r
data/ACDC-2D-LS/train_p_r_%: N=$(subst train_p_r_,,$(@F))
data/ACDC-2D-LS/train_p_r_%: results/learn_size/fs_% data/ACDC-2D-LS/train_r_%
	rm -rf $@_tmp $@
	$(CC) $(CFLAGS) inference.py --data_folder $(word 2, $^)/img --save_folder $@_tmp/tmp \
		--model_weights $</best.pkl --num_classes 4 --batch_size 10 $(DEBUG)
	mv $@_tmp/tmp/iter000 $@_tmp/gt
	rmdir $@_tmp/tmp
	cp -r $(word 2, $^)/img $@_tmp
	mv $@_tmp $@

results/learn_size/proposals_%: N=$(subst proposals_,,$(@F))
results/learn_size/proposals_%: OPT = --losses="[[('CrossEntropy', {'idc': [0, 3]}, None, None, None, 1)], \
			[('CrossEntropy', {'idc': [3]}, None, None, None, 1)]]" \
		--training_folders train_b_$(N) train_p_r_$(N) --val_loader_id 0
results/learn_size/proposals_%: DATA = --folders="[$(B_DATA)+[('gt', gt_transform, True)], $(B_DATA)+[('gt', gt_transform, True)]]"
results/learn_size/proposals_%: data/ACDC-2D-LS/val_20 data/ACDC-2D-LS/train_b_% data/ACDC-2D-LS/train_p_r_%
	rm -rf $@_tmp
	$(CC) $(CFLAGS) main.py --dataset=$(<D) --batch_size=1 --group --schedule \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=4 --metric_axis 3 \
		--grp_regex="$(G_RGX)" --validation_folder val_20 --in_memory --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@


# Plotting
results/learn_size/val_batch_dice.png results/learn_size/val_dice.png results/learn_size/val_haussdorf.png results/learn_size/tra_dice.png : COLS = 3
results/learn_size/tra_loss.png: COLS = 0
results/learn_size/val_dice.png results/learn_size/tra_loss.png results/learn_size/val_batch_dice.png results/learn_size/val_haussdorf.png: plot.py $(TRN)
results/learn_size/tra_dice.png : plot.py $(TRN)

results/learn_size/val_batch_dice_hist.png results/learn_size/val_dice_hist.png: COLS = 3
results/learn_size/tra_loss_hist.png: COLS = 0
results/learn_size/val_dice_hist.png results/learn_size/tra_loss_hist.png results/learn_size/val_batch_dice_hist.png: hist.py $(TRN)

results/learn_size/val_batch_dice_boxplot.png results/learn_size/val_dice_boxplot.png: COLS = 3
results/learn_size/val_batch_dice_boxplot.png results/learn_size/val_dice_boxplot.png: moustache.py $(TRN)

results/learn_size/%.png:
	$(eval metric:=$(subst _hist,,$(@F)))
	$(eval metric:=$(subst _boxplot,,$(metric)))
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless --epc 199 $(OPT) $(DEBUG)

metrics: $(TRN)
	$(CC) $(CFLAGS) metrics.py --num_classes=4 --grp_regex="$(G_RGX)" --gt_folder data/MIDL/val/gt \
		--pred_folders $(addsuffix /best_epoch/val, $^) $(DEBUG)


results/learn_size/abla_patient_avg.png: OPT= --mean_last --last_epc 50
results/learn_size/abla_patient%.png: plot_ablation_patients.py $(TRN)
	$(CC) $(CFLAGS) $< --headless --title "Dice over \# patients" --metric_axis 3 \
		--metric_logs $(addsuffix /val_batch_dice.npy, $(filter-out $<,$^)) --savefig $@ $(OPT) $(DEBUG)


# Viewing
view: $(TRN)
	viewer -n 3 --img_source data/ACDC-2D-LS/val_20/img data/ACDC-2D-LS/val_20/gt $(addsuffix /best_epoch/val, $^) --crop 30 \
		--display_names gt $(notdir $^) --remap "{1: 0, 2: 0}" $(DEBUG)

report: $(TRN)
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics val_batch_dice val_dice --axises 3 \
		--mode 'avg' --last_n_epc 50 --precision 3