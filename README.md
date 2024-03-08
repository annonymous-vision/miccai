# MICCAI annonymous

## Environment setup

Use the file **environment.yml** to create the environment.

## Unzip the code

All the code files are there in the src.zip file. Unzip it.

## Data Instructions

Download the VinDr and RSNA from the links for downstram evaluations:

- [RSNA](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
- [VinDr](vindr.ai/datasets/mammo)

## Prompts

The prompts in Fig.1 in Supplementary to synthesize the report are listed in the file:
`/src/breastclip/data/datasets/prompts.json`

## Preprocessing

To preprocess the VinDr and RSNA datasets using the steps discussed in the section 3.2 in the paper, follow:

```bash
python /src/Preprocessing/preprocess_in_house_dicom.py
```

To preprocess the bounding boxes of VinDr dataset, follow:

```bash
python /src/Preprocessing/preprocess_VinDr_detector.py
```

Update the dataset paths in the code.

## Train Mammo-CLIP

```bash
python /src/train.py --config-name src/configs/train_b5_det_in_house_wo_period_clip.yaml
```

## Zero-shot evaluation of Mammo-CLIP

```bash
FOLD=0
CKPT="model-best.tar"
DIR="dir/to/save/results"
FULL_CKPT="$DIR/checkpoints/fold_$FOLD/$CKPT"

python /src/eval_zero_shot_clip.py --config-name src/configs/zs_clip.yaml hydra.run.dir=$DIR model.clip_check_point=$FULL_CKPT
```

## Linear Probing of Mammo-CLIP on VinDR (Tab.1 in the paper)

* Adjust `--data_frac` to 0.1, 0.5, 1.0 for 10%, 50%, 100% of the dataset.

* Adjust `--label` to `Suspicious_Calcification, Mass, density` for the respective experiments.

```bash
python /src/train_classifier_RSNA.py \
  --img-dir 'img_directory_of_VinDr dataset' \
  --csv-file 'csv_file_location_of_VinDr dataset' \
  --data_frac 1.0 --dataset 'ViNDr' \
  --arch 'in_house_breast_clip_det_b5_period_n_lp' --label "Suspicious_Calcification" --epochs 20 --batch-size 4 --num-workers 0 \
  --clip_chk_pt_path "checkpoint path of Mammo-CLIP model" \
  --print-freq 10000 --log-freq 500 --running-interactive 'n' --n_folds 1 \
  --lr 5.0e-5 --weighted-BCE 'y' --balanced-dataloader 'n' 
```

## Finetuning of Mammo-CLIP on VinDR (Tab.1 in the paper)

```bash
python /src/train_classifier_RSNA.py \
  --img-dir 'img_directory_of_VinDr dataset' \
  --csv-file 'csv_file_location_of_VinDr dataset' \
  --data_frac 1.0 --dataset 'ViNDr' \
  --arch 'in_house_breast_clip_det_b5_period_n_ft' --label "Suspicious_Calcification" --epochs 20 --batch-size 4 --num-workers 0 \
  --clip_chk_pt_path "checkpoint path of Mammo-CLIP model" \
  --print-freq 10000 --log-freq 500 --running-interactive 'n' --n_folds 1 \
  --lr 5.0e-5 --weighted-BCE 'y' --balanced-dataloader 'n' 
```

## Finetuning of Mammo-CLIP on RSNA (Tab.2 in the paper)

* Adjust `--data_frac` to 0.1, 0.5, 1.0 for 10%, 50%, 100% of the dataset.

```bash
python /src/train_classifier_RSNA.py \
  --img-dir 'img_directory_of_RSNA dataset' \
  --csv-file 'csv_file_location_of_RSNA dataset' \
  --data_frac 1.0 --label "cancer" --n_folds 1 --lr 5e-5 --weight-decay 1e-4 --warmup-epochs 1 \
  --dataset 'RSNA' --arch 'in_house_breast_clip_det_b5_period_n_ft' --epochs 9 --batch-size 8 --num-workers 0 \
  --clip_chk_pt_path "checkpoint path of Mammo-CLIP model" \
  --print-freq 10000 --log-freq 500 --running-interactive 'n' \
  --lr 5.0e-5 --weighted-BCE 'y' --balanced-dataloader 'n'
```

## Linear Probing of Mammo-CLIP on RSNA (Tab.2 in the paper)

```bash
python /src/train_classifier_RSNA.py \
  --img-dir 'img_directory_of_RSNA dataset' \
  --csv-file 'csv_file_location_of_RSNA dataset' \
  --label "cancer" --n_folds 1 --lr 5e-5 --weight-decay 1e-4 --warmup-epochs 1 \
  --data_frac 1.0 --dataset 'RSNA' --arch 'in_house_breast_clip_det_b5_period_n_lp' --epochs 20 --batch-size 6 --num-workers 0 \
  --clip_chk_pt_path "checkpoint path of Mammo-CLIP model" \
  --print-freq 10000 --log-freq 500 --running-interactive 'n' \
  --lr 5.0e-5 --weighted-BCE 'y' --balanced-dataloader 'n'

```

## Localization of Mammo-CLIP on VinDr by finetuning the image encoder (Tab.3 in the paper)

* Adjust `--data_frac` to 0.1, 0.5, 1.0 for 10%, 50%, 100% of the dataset.

* Adjust `--concepts` to `Suspicious Calcification, Mass` for the respective experiments.

```bash
python /src/train_breast_concepts_detector.py \
  --img-dir 'img_directory_of_VinDr dataset' \
  --csv-file 'csv_file_location_of_VinDr dataset' \
  --dataset 'ViNDr' --arch 'clip_b5_in_house' --epochs 120 --batch-size 7 \
  --freeze_backbone "n" --data_frac 1.0 --concepts 'Mass' \
  --clip_chk_pt "checkpoint path of Mammo-CLIP model" \
  --print-freq 5000 --log-freq 300 --running-interactive 'n' --focal-alpha 0.25 --focal-gamma 2.0 \
  --inference-mode 'n' --score-threshold 0.2 --select-cancer 'y'
```

## Localization of Mammo-CLIP on VinDr by freezing the image encoder (Tab.3 in the paper)

```bash
python /src/train_breast_concepts_detector.py \
  --img-dir 'img_directory_of_VinDr dataset' \
  --csv-file 'csv_file_location_of_VinDr dataset' \
  --dataset 'ViNDr' --arch 'clip_b5_in_house' --epochs 120 --batch-size 7 \
  --freeze_backbone "y" --data_frac 1.0 --concepts 'Mass' \
  --clip_chk_pt "checkpoint path of Mammo-CLIP model" \
  --print-freq 5000 --log-freq 300 --running-interactive 'n' --focal-alpha 0.25 --focal-gamma 2.0 \
  --inference-mode 'n' --score-threshold 0.2 --select-cancer 'y'
```

### Train Mammo-Factor on VinDr

```bash
python /src/train_mapper.py \
  --arch "b5_clip_mapper" \
  --clip_chk_pt_path "checkpoint path of Mammo-CLIP model" \
  --attr_embs_path "path to the sentence embeddings corresponding to different attributes, e.g, mass, calcification etc. Sentences can be found in the /src/breastclip/data/datasets/prompts.json file"
``` 
