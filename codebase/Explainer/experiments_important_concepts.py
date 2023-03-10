import os
import sys

import numpy as np
import torch

import utils
from Explainer.utils_explainer import get_details_cubs_experiment, get_details_awa2_experiment, \
    get_details_ham10k_experiment, get_details_isic_experiment, get_details_cubs_experiment_baseline

sys.path.append(
    os.path.abspath("Path/PhD/miccai/codebase")
)


def get_concepts(main_args):
    for i in range(int(main_args.iter)):
        print(f"================>>>> Iteration: {i + 1} <<<<================")
        params = get_params(main_args, iteration=i + 1)
        train_mask_alpha = torch.BoolTensor()
        val_mask_alpha = torch.BoolTensor()
        test_mask_alpha = torch.BoolTensor()
        # train
        if main_args.dataset == "cub" or main_args.dataset == "HAM10k" or \
                main_args.dataset == "SIIM-ISIC" or main_args.dataset == "awa2":
            print("Getting train masks")
            _feature_names = [f"feature{j:010}" for j in range(params["train_tensor_concepts_bool"].size(1))]
            for _idx in range(params["train_tensor_concepts_bool"].size(0)):
                mask_alpha_norm = get_concept_masks(
                    _idx,
                    _feature_names,
                    params["train_tensor_y"],
                    params["train_tensor_preds"],
                    params["tensor_alpha_norm"],
                    params["train_tensor_concepts_bool"],
                    params["train_tensor_concepts"],
                    params["glt"]
                )
                train_mask_alpha = torch.cat((train_mask_alpha, mask_alpha_norm), dim=0)

        # val
        if main_args.dataset == "cub" or main_args.dataset == "HAM10k" or main_args.dataset == "SIIM-ISIC":
            print("Getting val masks")
            _feature_names = [f"feature{j:010}" for j in range(params["val_tensor_concepts_bool"].size(1))]
            for _idx in range(params["val_tensor_concepts_bool"].size(0)):
                mask_alpha_norm = get_concept_masks(
                    _idx,
                    _feature_names,
                    params["val_tensor_y"],
                    params["val_tensor_preds"],
                    params["tensor_alpha_norm"],
                    params["val_tensor_concepts_bool"],
                    params["val_tensor_concepts"],
                    params["glt"]
                )
                val_mask_alpha = torch.cat((val_mask_alpha, mask_alpha_norm), dim=0)

        # test
        if main_args.dataset == "cub" or main_args.dataset == "awa2":
            print("Getting test masks")
            _feature_names = [f"feature{j:010}" for j in range(params["test_tensor_concepts_bool"].size(1))]
            for _idx in range(params["test_tensor_concepts_bool"].size(0)):
                mask_alpha_norm = get_concept_masks(
                    _idx,
                    _feature_names,
                    params["test_tensor_y"],
                    params["test_tensor_preds"],
                    params["tensor_alpha_norm"],
                    params["test_tensor_concepts_bool"],
                    params["test_tensor_concepts"],
                    params["glt"]
                )
                test_mask_alpha = torch.cat((test_mask_alpha, mask_alpha_norm), dim=0)

        print(f"Train mask size: {train_mask_alpha.size()}")
        print(f"Val mask size: {val_mask_alpha.size()}")
        print(f"Test mask size: {test_mask_alpha.size()}")

        torch.save(train_mask_alpha, os.path.join(params["output_path"], f"train_mask_alpha.pt"))
        torch.save(val_mask_alpha, os.path.join(params["output_path"], f"val_mask_alpha.pt"))
        torch.save(test_mask_alpha, os.path.join(params["output_path"], f"test_mask_alpha.pt"))

        print(params["output_path"])


def get_concepts_baseline(main_args):
    params = get_params_baseline(main_args)
    train_mask_alpha = torch.BoolTensor()
    val_mask_alpha = torch.BoolTensor()
    test_mask_alpha = torch.BoolTensor()
    # train
    if main_args.dataset == "cub" or main_args.dataset == "HAM10k" or \
            main_args.dataset == "SIIM-ISIC" or main_args.dataset == "awa2":
        print("Getting train masks")
        _feature_names = [f"feature{j:010}" for j in range(params["train_tensor_concepts_bool"].size(1))]
        for _idx in range(params["train_tensor_concepts_bool"].size(0)):
            mask_alpha_norm = get_concept_masks(
                _idx,
                _feature_names,
                params["train_tensor_y"],
                params["train_tensor_preds"],
                params["tensor_alpha_norm"],
                params["train_tensor_concepts_bool"],
                params["train_tensor_concepts"],
                params["glt"]
            )
            train_mask_alpha = torch.cat((train_mask_alpha, mask_alpha_norm), dim=0)
    # val
    if main_args.dataset == "cub" or main_args.dataset == "HAM10k" or main_args.dataset == "SIIM-ISIC":
        print("Getting val masks")
        _feature_names = [f"feature{j:010}" for j in range(params["val_tensor_concepts_bool"].size(1))]
        for _idx in range(params["val_tensor_concepts_bool"].size(0)):
            mask_alpha_norm = get_concept_masks(
                _idx,
                _feature_names,
                params["val_tensor_y"],
                params["val_tensor_preds"],
                params["tensor_alpha_norm"],
                params["val_tensor_concepts_bool"],
                params["val_tensor_concepts"],
                params["glt"]
            )
            val_mask_alpha = torch.cat((val_mask_alpha, mask_alpha_norm), dim=0)
    # test
    if main_args.dataset == "cub" or main_args.dataset == "awa2":
        print("Getting test masks")
        _feature_names = [f"feature{j:010}" for j in range(params["test_tensor_concepts_bool"].size(1))]
        for _idx in range(params["test_tensor_concepts_bool"].size(0)):
            mask_alpha_norm = get_concept_masks(
                _idx,
                _feature_names,
                params["test_tensor_y"],
                params["test_tensor_preds"],
                params["tensor_alpha_norm"],
                params["test_tensor_concepts_bool"],
                params["test_tensor_concepts"],
                params["glt"]
            )
            test_mask_alpha = torch.cat((test_mask_alpha, mask_alpha_norm), dim=0)
    print(f"Train mask size: {train_mask_alpha.size()}")
    print(f"Val mask size: {val_mask_alpha.size()}")
    print(f"Test mask size: {test_mask_alpha.size()}")
    torch.save(train_mask_alpha, os.path.join(params["output_path"], f"train_mask_alpha.pt"))
    torch.save(val_mask_alpha, os.path.join(params["output_path"], f"val_mask_alpha.pt"))
    torch.save(test_mask_alpha, os.path.join(params["output_path"], f"test_mask_alpha.pt"))
    print(params["output_path"])


def get_concept_masks(
        idx,
        _feature_names,
        test_tensor_y,
        test_tensor_preds,
        tensor_alpha_norm,
        test_tensor_concepts_bool,
        test_tensor_concepts,
        model,
):
    device = utils.get_device()
    target_class = test_tensor_y[idx].to(torch.int32)
    y_hat = test_tensor_preds[idx].argmax(dim=0)
    percentile_selection = 99
    t = 0
    while True:
        if percentile_selection == 0:
            t = -1
            percentile_selection = 90
        percentile_val = np.percentile(
            tensor_alpha_norm[y_hat], percentile_selection
        )
        mask_alpha_norm = tensor_alpha_norm[y_hat] >= percentile_val
        mask = mask_alpha_norm

        # get the indexes of mask where the value is 1
        mask_indxs = (mask).nonzero(as_tuple=True)[0]
        imp_concepts = test_tensor_concepts_bool[idx][mask_indxs]
        imp_concept_vector = test_tensor_concepts[idx] * mask_alpha_norm
        y_pred_ex, _, _ = model(imp_concept_vector.unsqueeze(0).to(device))
        y_pred_ex = torch.nn.Softmax(dim=1)(y_pred_ex).argmax(dim=1)
        if y_pred_ex.item() == y_hat.item() or t == -1:
            break
        else:
            percentile_selection = percentile_selection - 1
    return mask_alpha_norm.reshape(1, mask_alpha_norm.size(0))


def get_concept_masks_top_k(
        idx,
        _feature_names,
        test_tensor_y,
        test_tensor_preds,
        tensor_alpha_norm,
        test_tensor_concepts_bool,
        test_tensor_concepts,
        model,
        n_concept_to_retain
):
    device = utils.get_device()
    target_class = test_tensor_y[idx].to(torch.int32)
    y_hat = test_tensor_preds[idx].argmax(dim=0)
    percentile_selection = 99
    t = 0
    mask_alpha_norm = torch.zeros(tensor_alpha_norm[y_hat].size())
    top_concepts = torch.topk(tensor_alpha_norm[y_hat], n_concept_to_retain)[1]
    concepts = test_tensor_concepts[idx]
    mask_alpha_norm[top_concepts] = True
    return mask_alpha_norm.reshape(1, mask_alpha_norm.size(0))


def get_params(main_args, iteration):
    if main_args.dataset == "cub" and main_args.arch == "ResNet101":
        return get_details_cubs_experiment(
            iteration, main_args.arch, alpha_KD=0.9, temperature_lens=0.7, layer="layer4"
        )
    elif main_args.dataset == "cub" and main_args.arch == "ViT-B_16":
        return get_details_cubs_experiment(iteration, main_args.arch, alpha_KD=0.99, temperature_lens=6.0, layer="VIT")
    elif main_args.dataset == "awa2" and main_args.arch == "ResNet50":
        lr = 0.001
        cov = 0.4
        prev_path = "cov_0.4_lr_0.001"
        return get_details_awa2_experiment(
            iteration, main_args.arch, alpha_KD=0.9, temperature_lens=0.7, lr=lr, cov=cov, layer="layer4",
            prev_path=prev_path
        )
    elif main_args.dataset == "awa2" and main_args.arch == "ViT-B_16":
        lr = 0.01
        cov = 0.2
        prev_path = "cov_0.2_lr_0.01"
        return get_details_awa2_experiment(
            iteration, main_args.arch, alpha_KD=0.99, temperature_lens=6.0, lr=lr, cov=cov, layer="VIT",
            prev_path=prev_path
        )
    elif main_args.dataset == "HAM10k":
        lr = 0.01
        cov = 0.2
        prev_path = "cov_0.2"
        return get_details_ham10k_experiment(
            iteration, main_args.arch, alpha_KD=0.9, temperature_lens=0.7, lr=lr, cov=cov,
            prev_path=prev_path
        )
    elif main_args.dataset == "SIIM-ISIC":
        lr = 0.01
        cov = 0.2
        prev_path = "cov_0.2"
        return get_details_isic_experiment(
            iteration, main_args.arch, alpha_KD=0.9, temperature_lens=0.7, lr=lr, cov=cov,
            prev_path=prev_path
        )


def get_params_baseline(main_args):
    if main_args.dataset == "cub" and main_args.arch == "ViT-B_16":
        path = "Path/PhD/miccai/out/cub/Baseline_PostHoc/ViT-B_16/lr_0.01_epochs_500_temperature-lens_6.0_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_VIT_explainer_init_none"
        return get_details_cubs_experiment_baseline(path)
    elif main_args.dataset == "awa2" and main_args.arch == "ResNet50":
        pass
    elif main_args.dataset == "awa2" and main_args.arch == "ViT-B_16":
        pass
    elif main_args.dataset == "HAM10k":
        pass
    elif main_args.dataset == "SIIM-ISIC":
        pass
