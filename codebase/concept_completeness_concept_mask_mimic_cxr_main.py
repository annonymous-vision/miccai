import argparse
import json
import os
import shutil
import sys

import torch

import Completeness_and_interventions.concept_completeness_intervention_utils as cci
import MIMIC_CXR.mimic_cxr_utils as mimic_utils

sys.path.append(os.path.abspath("Path/PhD/miccai"))


def config():
    parser = argparse.ArgumentParser(description='Get important concepts masks')
    parser.add_argument('--base_path', metavar='DIR',
                        default='Path/PhD/miccai',
                        help='path to checkpoints')
    parser.add_argument('--output', metavar='DIR',
                        default='Path/PhD/miccai/out',
                        help='path to output logs')

    parser.add_argument('--disease', type=str, default="effusion", help='dataset name')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument('--iterations', type=int, default="1", help='total number of iteration')
    parser.add_argument('--top_K', nargs='+', default=[3, 5, 10, 15, 20, 25, 30, 50], type=int,
                        help='How many concepts?')
    parser.add_argument('--model', default='MoIE', type=str, help='MoIE or Baseline_CBM_logic or Baseline_PCBM_logic')
    parser.add_argument('--icml', default='n', type=str, help='ICML or MICCAI')

    return parser.parse_args()


def create_dataset(args, output, json_file, dataset_path, save_path):
    args.metric = "auroc"
    print(f"############################### Creating master df ###############################")
    mimic_utils.merge_csv_from_experts(args.iterations, json_file, args.disease, output, save_path, mode="test")
    mimic_utils.merge_csv_from_experts(args.iterations, json_file, args.disease, output, save_path, mode="train")
    mimic_utils.merge_csv_from_experts(args.iterations, json_file, args.disease, output, save_path, mode="val")
    print(f"##################################################################################")

    save_path_top_K = os.path.join(save_path, f"concepts_topK_{args.topK}")
    os.makedirs(save_path_top_K, exist_ok=True)
    print(f"############################### Creating test data master df ###############################")

    mimic_utils.get_expert_specific_outputs(
        args.iterations, args, json_file, output, dataset_path, save_path_top_K, mode="test"
    )
    mimic_utils.get_expert_specific_outputs(
        args.iterations, args, json_file, output, dataset_path, save_path_top_K, mode="train"
    )
    mimic_utils.get_expert_specific_outputs(
        args.iterations, args, json_file, output, dataset_path, save_path_top_K, mode="val"
    )


def create_dataset_for_completeness_moIE(args):
    _disease = args.disease
    _iters = args.iterations
    _seed = args.seed
    _output = args.output
    if args.icml == "y":
        print("=====================>>>>> Completeness score masks for ICML paper <<<<<<=======================")
        dataset_path = f"{_output}/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/{_disease}/dataset_g"
        output = f"Path/PhD/miccai/out/mimic_cxr/explainer/{_disease}"
        save_path = os.path.join(
            args.output, "mimic_cxr", "completeness_icml", "dataset", "moIE", _disease
        )
        json_file = os.path.join(args.base_path, "codebase", "MIMIC_CXR", "paths_mimic_cxr_icml.json")
    else:
        dataset_path = f"{_output}/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/{_disease}/dataset_g"
        output = f"{_output}/mimic_cxr/soft_concepts/seed_{_seed}/explainer/{_disease}"
        save_path = os.path.join(
            args.output, "mimic_cxr", "completeness", f"seed_{_seed}", "dataset", "moIE", _disease
        )
        json_file = os.path.join(args.base_path, "codebase", "MIMIC_CXR", "paths_mimic_cxr.json")

    os.makedirs(save_path, exist_ok=True)
    create_dataset(args, output, json_file, dataset_path, save_path)


def save_baseline_output(args, output_path, dataset_path, save_path, mode):
    (
        _, tensor_alpha_norm, _, mask_by_pi, out_put_g_pred, out_put_bb_pred, out_put_target,
        proba_concepts, ground_truth_concepts, _
    ) = mimic_utils.get_outputs(1, args, output_path, dataset_path, mode=mode)

    mask_alpha = cci.get_concept_masks_top_k(
        out_put_g_pred, tensor_alpha_norm, proba_concepts, args.topK
    )

    torch.save(mask_alpha, os.path.join(save_path, f"{mode}_all_mask_alpha.pt"))
    torch.save(out_put_g_pred, os.path.join(save_path, f"{mode}_all_preds_g.pt"))
    torch.save(out_put_bb_pred, os.path.join(save_path, f"{mode}_all_preds_bb.pt"))
    torch.save(out_put_target, os.path.join(save_path, f"{mode}_all_ground_truth_labels.pt"))
    torch.save(proba_concepts, os.path.join(save_path, f"{mode}_all_proba_concepts.pt"))
    torch.save(ground_truth_concepts, os.path.join(save_path, f"{mode}_all_ground_truth_concepts.pt"))


def create_dataset_for_completeness_baselines(args):
    _disease = args.disease
    _iters = args.iterations
    _seed = args.seed
    _output = args.output
    args.metric = "auroc"

    save_path = os.path.join(
        args.output, "mimic_cxr", "completeness", f"seed_{_seed}", "dataset", args.model, _disease,
        f"concepts_topK_{args.topK}"
    )
    os.makedirs(save_path, exist_ok=True)

    json_file = os.path.join(args.base_path, "codebase", "MIMIC_CXR", "paths_mimic_cxr.json")
    with open(json_file) as _file:
        paths = json.load(_file)

    output_path = None
    dataset_path = None
    if args.model == "Baseline":
        dataset_path = f"Path/PhD/miccai/out/mimic_cxr/Baseline/Backbone/lr_0.01_epochs_60/densenet121/{args.disease}/dataset_g"
        root = paths[args.disease]["baseline_cbm_logic_paths"]["base_path"]
        output_path = f"{_output}/mimic_cxr/{root}/densenet121/{args.disease}/seed_{args.seed}"
    elif args.model == "Baseline_PostHoc":
        dataset_path = f"{_output}/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/{_disease}/dataset_g"
        root = paths[args.disease]["baseline_pcbm_logic_paths"]["base_path"]
        output_path = f"{_output}/mimic_cxr/Baseline_PostHoc/densenet121/{args.disease}/{root}/seed_{args.seed}"
        src_train = os.path.join(args.output, "mimic_cxr", f"Baseline_PostHoc/densenet121/{args.disease}", root,
                                 f"seed_{_seed}", f"train_FOL_results_baseline_pcbm.csv")
        src_test = os.path.join(args.output, "mimic_cxr", f"Baseline_PostHoc/densenet121/{args.disease}", root,
                                f"seed_{_seed}", f"test_FOL_results_baseline_pcbm.csv")
        dst_train = os.path.join(args.output, "mimic_cxr", "completeness", f"seed_{_seed}", "dataset/Baseline_PostHoc",
                                 args.disease, f"train_master_FOL_results.csv")
        dst_test = os.path.join(args.output, "mimic_cxr", "completeness", f"seed_{_seed}", "dataset/Baseline_PostHoc",
                                args.disease, f"test_master_FOL_results.csv")
        shutil.copyfile(src_test, dst_test)
        shutil.copyfile(src_train, dst_train)

    save_baseline_output(args, output_path, dataset_path, save_path, mode="train")
    save_baseline_output(args, output_path, dataset_path, save_path, mode="test")


def main():
    args = config()
    print(f"Get important concept masks:")
    for top_k in args.top_K:
        args.topK = top_k
        if args.model == "MoIE":
            create_dataset_for_completeness_moIE(args)
        # elif args.model == "Baseline":
        #     create_dataset_for_completeness_baselines(args, model_type="Baseline")
        elif args.model == "Baseline_PostHoc":
            create_dataset_for_completeness_baselines(args)


if __name__ == '__main__':
    main()
