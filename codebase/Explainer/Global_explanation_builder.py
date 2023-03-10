import os
import pickle
import time

import numpy as np
import sympy
import torch
from sklearn.metrics import f1_score, accuracy_score
from sympy import simplify_logic
from sympy import to_dnf, lambdify
from torch.nn.functional import one_hot
from tqdm import tqdm

from utils import replace_names


def FOL_complexity(formula, to_dnf: bool = False):
    """
    Estimates the complexity of the formula.

    :param formula: logic formula.
    :param to_dnf: whether to convert the formula in disjunctive normal form.
    :return: The complexity of the formula.
    """
    if formula != "" and formula is not None:
        if to_dnf:
            formula = str(sympy.to_dnf(formula))
        return np.array([len(f.split(" & ")) for f in formula.split(" | ")]).sum()
    return 0


def test_explanation(formula: str, x: torch.Tensor, y: torch.Tensor, target_class: int):
    """
    Tests a logic formula.

    :param formula: logic formula
    :param x: input data
    :param y: input labels (MUST be one-hot encoded)
    :param target_class: target class
    :return: Accuracy of the explanation and predictions
    """

    y = y[:, target_class]
    concept_list = [f"feature{i:010}" for i in range(x.shape[1])]
    # get predictions using sympy
    explanation = to_dnf(formula)
    fun = lambdify(concept_list, explanation, "numpy")
    x = x.cpu().detach().numpy()
    predictions = fun(*[x[:, i] > 0.5 for i in range(x.shape[1])])
    # get accuracy
    if y.size(0) != 0:
        f1 = f1_score(y, predictions)
        acc = accuracy_score(y, predictions)
    else:
        f1 = 0
        acc = 0

    return f1, acc, predictions, y


def get_local_results(
        idx, pkl, feature_names, tensor_y, tensor_preds, tensor_alpha_norm, tensor_concepts_bool, tensor_concepts,
        moIE, device,
):
    target_class = tensor_y[idx].to(torch.int32)
    y_hat = tensor_preds[idx].argmax(dim=0)
    percentile_selection = 99
    ps = 100
    while True:
        if percentile_selection == 0:
            percentile_selection = 80
            ps = 0
        percentile_val = np.percentile(
            tensor_alpha_norm[y_hat], percentile_selection
        )
        mask_alpha_norm = tensor_alpha_norm[y_hat] >= percentile_val
        mask = mask_alpha_norm
        # get the indexes of mask where the value is 1
        mask_indxs = (mask).nonzero(as_tuple=True)[0]
        imp_concept_vector = tensor_concepts[idx] * mask_alpha_norm
        y_pred_ex, _, _ = moIE(imp_concept_vector.unsqueeze(0).to(device))
        y_pred_ex = torch.nn.Softmax(dim=1)(y_pred_ex).argmax(dim=1)
        if ps == 0:
            break
        if y_pred_ex.item() == y_hat.item():
            break
        else:
            percentile_selection = percentile_selection - 1

    dict_sample_concept = {}
    for concept in pkl.concept_names:
        dict_sample_concept[concept] = 0

    dict_sample_concept["y_GT"] = 0
    dict_sample_concept["y_BB"] = 0
    dict_sample_concept["y_G"] = 0

    concepts = []
    for indx in mask_indxs:
        dict_sample_concept[pkl.concept_names[indx]] = 1
        concepts.append(pkl.concept_names[indx])
    dict_sample_concept["y_GT"] = target_class.item()
    dict_sample_concept["y_G"] = y_hat.item()
    dict_sample_concept["correctly_predicted_wrt_GT"] = (target_class.item() == y_hat.item())

    # concept_label.append(dict_sample_concept)
    explanations = ""
    for m_idx in mask_indxs.tolist():
        if explanations:
            explanations += " & "

        if tensor_concepts_bool[idx][m_idx] == 0:
            explanations += f"~{feature_names[m_idx]}"
        elif tensor_concepts_bool[idx][m_idx] == 1:
            explanations += f"{feature_names[m_idx]}"

    explanation_complete = replace_names(explanations, pkl.concept_names)

    # For debugging
    # print(f"{idx}: ==================================================>")
    # print(f"Ground Truth class_label: {pkl.labels[target_class]} ({target_class})")
    # print(f"Predicted(g) class_label: {pkl.labels[y_hat]} ({y_hat})")
    # print("Concept Explanations: =======================>>>>")
    # print(f"{pkl.labels[y_hat]} ({y_hat}) <=> {explanation_complete}")

    return {
        "dict_sample_concept": dict_sample_concept,
        "num_concepts": len(concepts),
        # "concept_dict_key": int(y_hat.item()) if (target_class.item() == y_hat.item()) else -1,
        "concept_dict_key": int(y_hat.item()),
        "concept_dict_val": explanation_complete,
        "test_tensor_concepts": tensor_concepts[idx],
        "correctly_predicted": (target_class == y_hat),
        "raw_explanations": explanations
    }


def get_local_FOL(pkl, tensor_y, tensor_preds, tensor_alpha_norm, tensor_concepts_bool, tensor_concepts, moIE, device):
    _feature_names = [f"feature{j:010}" for j in range(tensor_concepts_bool.size(1))]
    num_concepts_ex = []
    results_arr = []
    with tqdm(total=tensor_concepts_bool.size(0)) as t:
        for _idx in range(tensor_concepts_bool.size(0)):
            results = get_local_results(
                _idx, pkl, _feature_names, tensor_y, tensor_preds, tensor_alpha_norm, tensor_concepts_bool,
                tensor_concepts, moIE, device,
            )
            results_arr.append(results)
            num_concepts_ex.append(results["num_concepts"])
            t.set_postfix(sample_id='{0}'.format(_idx))
            t.update()

    return results_arr


def polish_local_FOL(y_unique, concept_dict, test_tensor_concepts, test_tensor_y, test_tensor_preds, top_K_support):
    expl_dict = {}
    for y in y_unique:
        expl_dict[int(y.item())] = []

    for target_class in list(concept_dict.keys()):
        expl = []
        for ex in concept_dict[target_class]:
            ids = (test_tensor_y == target_class).nonzero(as_tuple=True)[0]
            y_target = test_tensor_y[ids]
            y_target1h = one_hot(
                y_target.to(torch.long), num_classes=test_tensor_preds.size(1)
            ).to(torch.float)
            x_target = test_tensor_concepts[ids]
            if not ex["test"]:
                f1, acc, predictions, y = test_explanation(ex["raw_explanations"], x_target, y_target1h, target_class)
            else:
                f1 = 0
                acc = 0
            expl.append(
                {
                    "raw_explanations": ex["raw_explanations"],
                    "explanations": ex["explanations"],
                    "f1": f1,
                    "acc": acc,
                }
            )

        sorted_list = sorted(expl, key=lambda d: d["acc"], reverse=True)[:top_K_support]
        res = list(set(dic["raw_explanations"] for dic in sorted_list))
        expl_dict[target_class] = res

    return expl_dict


def create_global_explanations(expl_dict, pkl, test_tensor_y, test_tensor_preds, test_tensor_concepts):
    global_explanations = {}
    y_GT = torch.FloatTensor()
    y_fol_pred = []
    fol_complexity = []
    keys = list(expl_dict.keys())
    for i in keys:
        print("================")
        print(f"{pkl.labels[i]} ({i})")
        print("================")
        explanations = []
        aggregated_explanation = None
        for explanation_raw in expl_dict[i]:
            explanations.append(explanation_raw)

            # aggregate example-level explanations
            # print(explanation_raw)
            # print(explanations)
            aggregated_explanation = " | ".join(explanations)

        print(aggregated_explanation)
        global_expl_raw = simplify_logic(aggregated_explanation, "dnf", force=True)
        global_expl = replace_names(str(global_expl_raw), pkl.concept_names)
        ids = (test_tensor_y == i).nonzero(as_tuple=True)[0]
        y_target = test_tensor_y[ids]
        y_target1h = one_hot(
            y_target.to(torch.long), num_classes=test_tensor_preds.size(1)
        ).to(torch.float)
        x_target = test_tensor_concepts[ids]
        f1, acc, predictions, y_t = test_explanation(
            global_expl_raw,
            x_target,
            y_target1h,
            i,
        )
        y_fol_pred.append(predictions)
        y_GT = torch.cat((y_GT, y_t), dim=0)
        complexity = FOL_complexity(global_expl)
        fol_complexity.append(complexity)
        print(global_expl)
        global_explanations[pkl.labels[i]] = {
            "class": pkl.labels[i],
            "local_explanations": aggregated_explanation,
            "global_explanation_raw": str(global_expl_raw),
            "global_explanation": global_expl,
            "complexity": complexity,
            "f1": f1,
            "acc": acc
        }

    y_fol_pred = np.concatenate(y_fol_pred, axis=0)
    y_GT = y_GT.cpu().numpy()
    FOL_F1 = f1_score(y_GT, y_fol_pred)
    FOL_acc = accuracy_score(y_GT, y_fol_pred)
    mean_complexity = np.array(fol_complexity)
    print("\n>>>>>>>>>>>>>>> Statistics <<<<<<<<<<<<<<<")
    print(f"FOL_F1: {FOL_F1}, FOL_acc: {FOL_acc}, mean_complexity:{np.mean(mean_complexity)}")
    return {
        "global_explanations": global_explanations,
        "FOL_F1": FOL_F1,
        "FOL_acc": FOL_acc,
        "FOL_complexity": np.mean(mean_complexity)
    }


def build_FOL(_dict, args):
    train_tensor_y = _dict["train_tensor_y"]
    train_tensor_preds = _dict["train_tensor_preds"]
    train_tensor_alpha_norm = _dict["tensor_alpha_norm"]
    train_tensor_concepts_bool = _dict["train_tensor_concepts_bool"]
    train_tensor_concepts = _dict["train_tensor_concepts"]
    test_tensor_concepts = _dict["test_tensor_concepts"]
    test_tensor_y = _dict["test_tensor_y"]
    test_tensor_preds = _dict["test_tensor_preds"]
    test_tensor_concepts_bool = _dict["test_tensor_concepts_bool"]
    val_tensor_concepts = _dict["val_tensor_concepts"]
    val_tensor_y = _dict["val_tensor_y"]
    val_tensor_preds = _dict["val_tensor_preds"]
    val_tensor_concepts_bool = _dict["val_tensor_concepts_bool"]

    moIE = _dict["moIE"]
    pkl = _dict["pkl"]
    device = _dict["device"]
    start = time.time()

    results_arr = get_local_FOL(
        pkl, train_tensor_y, train_tensor_preds, train_tensor_alpha_norm, train_tensor_concepts_bool,
        train_tensor_concepts, moIE, device
    )
    done = time.time()
    elapsed = done - start
    print("Time get the local FOL: " + str(elapsed) + " secs")

    concept_label = []
    num_concepts_ex = []
    concept_dict = {}
    y_unique = torch.unique(test_tensor_y)
    for y in y_unique:
        concept_dict[int(y.item())] = []
    for results in results_arr:
        concept_label.append(results["dict_sample_concept"])
        num_concepts_ex.append(results["num_concepts"])
        # if results["concept_dict_key"] != -1:
        if results["concept_dict_key"] in concept_dict:
            concept_dict[results["concept_dict_key"]].append(
                {
                    "explanations": results["concept_dict_val"],
                    "raw_explanations": results["raw_explanations"],
                    "test": False
                }
            )

    # It may happen that the target class may not be a part of training. For ex, no samples of class
    # ``Bay_breasted_warbler'' may exist in training. In that case the global FOL will be simply
    # the OR(AND(local_FOLs)) without any validation support.
    keys = [k for k, v in concept_dict.items() if len(v) == 0]
    print(keys)
    for k in keys:
        print(k)
        ids = (test_tensor_y == k).nonzero(as_tuple=True)[0]
        print(ids)
        y_target = test_tensor_y[ids]
        x_target = test_tensor_concepts[ids]
        concepts_bool_target = test_tensor_concepts_bool[ids]
        preds_target = test_tensor_preds[ids]
        results_arr = get_local_FOL(
            pkl, y_target, preds_target, train_tensor_alpha_norm, concepts_bool_target,
            x_target, moIE, device
        )
        # print(results_arr)
        for results in results_arr:
            concept_label.append(results["dict_sample_concept"])
            num_concepts_ex.append(results["num_concepts"])
            # if results["concept_dict_key"] != -1:

            concept_dict[k].append(
                {
                    "explanations": results["concept_dict_val"],
                    "raw_explanations": results["raw_explanations"],
                    "test": True
                }
            )
    keys = [k for k, v in concept_dict.items() if len(v) == 0]
    print(keys)
    print("\nPolishing local FOLs based on support")
    expl_dict = polish_local_FOL(
        y_unique, concept_dict, val_tensor_concepts, val_tensor_y, val_tensor_preds, args.top_K_support
    )

    # print("\nBuilding global FOL per class")
    # results = create_global_explanations(expl_dict, pkl, test_tensor_y, test_tensor_preds, test_tensor_concepts)
    os.makedirs(os.path.join(args.save_path, f"{args.dataset}-{args.arch}"), exist_ok=True)
    pickle.dump(
        results_arr,
        open(os.path.join(
            args.save_path, f"{args.dataset}-{args.arch}", f"explanations-save-expert{args.cur_iter}.pkl"), "wb"
        )
    )

    pickle.dump(
        num_concepts_ex,
        open(os.path.join(
            args.save_path, f"{args.dataset}-{args.arch}", f"num-concepts-expert{args.cur_iter}.pkl"
        ), "wb")
    )

    pickle.dump(
        expl_dict,
        open(os.path.join(args.save_path, f"{args.dataset}-{args.arch}", f"FOLs-expert{args.cur_iter}.pkl"), "wb")
    )

    print("\n>>>>>>>>>>>>>>>>>>>>> Saved Paths <<<<<<<<<<<<<<<<<<<<<<")
    print(
        f"Big explanation array is saved at: "
        f"{args.save_path}/{args.dataset}-{args.arch}/explanations-save-expert{args.cur_iter}.pkl"
    )

    print(
        f"Big concept array is saved at: "
        f"{args.save_path}/{args.dataset}-{args.arch}/num-concepts-expert{args.cur_iter}.pkl"
    )
    print(f"All FOLs is saved at: {args.save_path}/{args.dataset}-{args.arch}/FOLs-expert{args.cur_iter}.pkl")
