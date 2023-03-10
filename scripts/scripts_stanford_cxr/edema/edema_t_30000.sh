#!/bin/sh
#SBATCH --output=Path/Stanford_CXR/t/edema_30000_%j.out
pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")
echo $CURRENT

slurm_output=Path/Stanford_CXR/t/edema_30000_$CURRENT.out

echo "Stanford_CXR"
source Path/anaconda3/etc/profile.d/conda.sh
# conda activate python_3_7

conda activate python_3_7_rtx_6000

python Path/PhD/miccai/codebase/train_t_ssl_main.py \
  --target_dataset "stanford_cxr" \
  --source_dataset "mimic_cxr" \
  --disease "edema" \
  --source-checkpoint-bb_path "lr_0.01_epochs_60_loss_CE" \
  --source-checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --source-checkpoint-t-path "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --source-checkpoint-t "g_best_model_epoch_8.pth.tar" \
  --tot_samples 30000 >$slurm_output


python Path/PhD/miccai/codebase/test_t_ssl_main.py \
  --target_dataset "stanford_cxr" \
  --source_dataset "mimic_cxr" \
  --disease "edema" \
  --source-checkpoint-bb_path "lr_0.01_epochs_60_loss_CE" \
  --source-checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --source-checkpoint-t-path "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --source-checkpoint-t "g_best_model_epoch_8.pth.tar" \
  --tot_samples 30000 >$slurm_output