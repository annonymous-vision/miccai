# MICCAI annonymous

## Environment setup

Use the file **environment.yml** to create the environment.

## Data Instructions

Below we list the data sources to download the datasets we used to train / evaluate MoIEs. Once you download a dataset,
update the correct paths in the variable `--data-root` for the files containing `main` function in `./codebase/<files>`.

### Data Requirements and Download Links

- [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
- [RadGraph](https://physionet.org/content/radgraph/1.0.0/)
- [NVIDIA Annotation](https://github.com/leotam/MIMIC-CXR-annotations)
- [Chest ImaGenome](https://physionet.org/content/chest-imagenome/1.0.0/)

### Data Preprocessing

1. Run `./preprocessing/radgraph_itemized.py` to generate itemized RadGraph examples.
2. Run `./preprocessing/radgraph_parsed.py` to parse RadGraph relations.
3. Run `./preprocessing/adj_matrix.py` to create adjacency matrix that represents the relations between anatomical
   landmarks and observations mentioned in radiology reports. These will be the concepts for training MoIE-CXR.

## Running MoIE-CXR for MIMIC-CXR

* Go the `./scripts/scripts_mimic_cxr` folder and get the training scripts. For all the diseases in for **MIMIC-CXR**,
  one script is included in the respective folder with proper instructions to run **1) Blackbox 2) projection (t) 3)
  interpretable model (g) 4) residual (r)**. **For example, to run `cardiomegaly` for **MIMIC-CXR**, look into the
  file `./scripts/scripts_mimic_cxr/cardiomegaly/cardiomegaly.sh`**
* The naming convention and the paths to be replaced is mentioned in the script. Follow them carefully
* Run them sequentially.
* Due to anonymity, we can not upload the pretrained models. Upon decision, we will upload the pretrained model as well.

## FOLs for MIMIC-CXR

We include the results of for every samples in the test-set of MIMIC-CXR in the csv
file `./results/<disease>/test_results_expert_<id>.csv`. For example to see the results of expert1 for cardiomegaly,
refer to the column `actual_explanation` in the file: `./results/cardiomegaly/test_results_expert_1.csv`

## Finetune for Stanford-CXR

* Go the `./scripts/scripts_stanford_cxr` folder and get the training scripts. For all the diseases in for **
  Stanford-CXR**, one script is included in the respective folder with proper instructions to run **1) Blackbox 2)
  projection (t) 3)
  interpretable model (g) 4) residual (r)**. Fix the number of samples to be used as training data for Stanford-CXR. **
  For example, to run `cardiomegaly` for **Stanford-CXR**, look into the
  file `./scripts/scripts_stanford_cxr/cardiomegaly/car_15000.sh`
  and `./scripts/scripts_stanford_cxr/cardiomegaly/car_fl_15000.sh` to finetune the model and estimate the computation
  cost.** These two files use 15000 samples of Stanford-CXR. To modify this number, use the following variable `--tot_samples` in
  those scripts.
* The naming convention and the paths to be replaced is mentioned in the script. Follow them carefully
* Run them sequentially.
* Due to anonymity, we can not upload the pretrained models. Upon decision, we will upload the pretrained model as well.

