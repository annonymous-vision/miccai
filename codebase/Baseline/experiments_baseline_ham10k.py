import logging
import os
import pickle
import random
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torchvision.transforms import transforms
from tqdm import tqdm

import utils
from Baseline.models.baseline_HAM10k import BaseLine_HAM10k
from Explainer.utils_explainer import ConceptBank
from Logger.logger_mimic_cxr import Logger_MIMIC_CXR
from dataset.dataset_ham10k import load_ham_data

logger = logging.getLogger(__name__)


def test_explainer(args):
    print("###############################################")
    print("Testing baseline: HAM10k")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = "lr_0.01_epochs_250"
    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "Baseline", root, "explainer")
    output_path = os.path.join(args.output, args.dataset, "Baseline", root, "explainer")

    print("########### Paths ###########")
    print(chk_pt_path)
    print(output_path)
    print("########### Paths ###########")
    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    device = utils.get_device()
    print(f"Device: {device}")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize
        ]
    )

    pickle.dump(args, open(os.path.join(output_path, "test_explainer_configs.pkl"), "wb"))
    train_loader, val_loader, idx_to_class = load_ham_data(args, transform, args.class_to_idx, mode="save")
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    concepts_dict = pickle.load(open(args.concept_file_name, "rb"))
    concept_bank = ConceptBank(concepts_dict, device)
    glt_chk_pt = os.path.join(chk_pt_path, args.checkpoint_model)
    baseline = BaseLine_HAM10k(args, concept_bank).to(device)
    baseline.load_state_dict(torch.load(glt_chk_pt)["state_dict"])
    baseline.eval()

    print("Save overall whole model outputs")
    predict(
        baseline,
        val_loader,
        concept_bank,
        output_path,
        device,
        mode="val"
    )

    print("!! Saving train loader only selected by g!!")


def predict(
        baseline,
        val_loader,
        concept_bank,
        output_path,
        device,
        mode
):
    tensor_ori_images = torch.FloatTensor()
    tensor_images = torch.FloatTensor()
    tensor_concepts = torch.FloatTensor().cuda()
    tensor_preds = torch.FloatTensor().cuda()
    tensor_y = torch.FloatTensor().cuda()
    tensor_conceptizator_concepts = torch.FloatTensor().cuda()
    # tensor_conceptizator_threshold = torch.FloatTensor().cuda()
    tensor_concept_mask = torch.FloatTensor().cuda()
    tensor_alpha = torch.FloatTensor().cuda()
    tensor_alpha_norm = torch.FloatTensor().cuda()

    with torch.no_grad():
        with tqdm(total=len(val_loader)) as t:
            for batch_id, (image, raw_image, label) in enumerate(val_loader):
                raw_image = raw_image.to(device)
                image, label = image.to(device), label.to(torch.long).to(device)
                y_hat, concepts, concept_mask, \
                alpha, alpha_norm, conceptizator = baseline(image, test=True)

                tensor_ori_images = torch.cat((tensor_ori_images, raw_image.cpu()), dim=0)
                tensor_images = torch.cat((tensor_images, image.cpu()), dim=0)
                tensor_concepts = torch.cat((tensor_concepts, concepts), dim=0)
                tensor_preds = torch.cat((tensor_preds, y_hat), dim=0)
                tensor_y = torch.cat((tensor_y, label), dim=0)
                tensor_conceptizator_concepts = torch.cat(
                    (tensor_conceptizator_concepts, conceptizator.concepts), dim=1
                )

                # tensor_conceptizator_threshold = conceptizator.threshold
                tensor_concept_mask = concept_mask
                tensor_alpha = alpha
                tensor_alpha_norm = alpha_norm
                t.set_postfix(batch_id="{0}".format(batch_id))
                t.update()

    tensor_concepts = tensor_concepts.cpu()
    tensor_preds = tensor_preds.cpu()
    tensor_y = tensor_y.cpu()
    tensor_conceptizator_concepts = tensor_conceptizator_concepts.cpu()
    # tensor_conceptizator_threshold = tensor_conceptizator_threshold.cpu()
    tensor_concept_mask = tensor_concept_mask.cpu()
    tensor_alpha = tensor_alpha.cpu()
    tensor_alpha_norm = tensor_alpha_norm.cpu()

    print("Output sizes: ")
    print(f"tensor_ori_images size: {tensor_ori_images.size()}")
    print(f"tensor_images size: {tensor_images.size()}")
    print(f"tensor_concepts size: {tensor_concepts.size()}")
    print(f"tensor_preds size: {tensor_preds.size()}")
    print(f"tensor_y size: {tensor_y.size()}")
    print(f"tensor_conceptizator_concepts size: {tensor_conceptizator_concepts.size()}")

    print("Model-specific sizes: ")
    # print(f"tensor_conceptizator_threshold: {tensor_conceptizator_threshold}")
    print(f"tensor_concept_mask size: {tensor_concept_mask.size()}")
    print(f"tensor_alpha size: {tensor_alpha.size()}")
    print(f"tensor_alpha_norm size: {tensor_alpha_norm.size()}")

    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_ori_images.pt"), tensor_to_save=tensor_ori_images
    )

    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_images.pt"), tensor_to_save=tensor_images
    )

    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_concepts.pt"), tensor_to_save=tensor_concepts
    )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_preds.pt"), tensor_to_save=tensor_preds
    )

    utils.save_tensor(path=os.path.join(output_path, f"{mode}_tensor_y.pt"), tensor_to_save=tensor_y)
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_conceptizator_concepts.pt"),
        tensor_to_save=tensor_conceptizator_concepts
    )

    # utils.save_tensor(
    #     path=os.path.join(output_path, f"{mode}_tensor_conceptizator_threshold.pt"),
    #     tensor_to_save=tensor_conceptizator_threshold
    # )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_concept_mask.pt"),
        tensor_to_save=tensor_concept_mask
    )

    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_alpha.pt"),
        tensor_to_save=tensor_alpha
    )
    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_tensor_alpha_norm.pt"), tensor_to_save=tensor_alpha_norm
    )


def train_explainer(args):
    print("###############################################")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = f"lr_{args.lr}_epochs_{args.epochs}"
    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "Baseline", root, "explainer")
    output_path = os.path.join(args.output, args.dataset, "Baseline", root, "explainer")
    tb_logs_path = os.path.join(args.logs, args.dataset, "Baseline", f"{root}_explainer")

    print("########### Paths ###########")
    print(chk_pt_path)
    print(output_path)
    print(tb_logs_path)
    print("########### Paths ###########")
    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)
    device = utils.get_device()
    print(f"Device: {device}")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize
        ]
    )
    train_loader, val_loader, idx_to_class = load_ham_data(args, transform, args.class_to_idx)
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    concepts_dict = pickle.load(open(args.concept_file_name, "rb"))
    concept_bank = ConceptBank(concepts_dict, device)
    baseline = BaseLine_HAM10k(args, concept_bank).to(device)
    solver = torch.optim.SGD(baseline.parameters(), lr=args.lr_explainer, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    final_parameters = OrderedDict(
        arch=[args.arch],
        dataset=[args.dataset],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')]
    )

    run_id = utils.get_runs(final_parameters)[0]
    best_auroc = 0
    logger = Logger_MIMIC_CXR(
        1, best_auroc, 0, chk_pt_path, tb_logs_path, output_path, train_loader, val_loader, len(args.labels),
        model_type="bb"
    )

    fit_explainer(
        args.epochs,
        baseline,
        criterion,
        solver,
        train_loader,
        val_loader,
        logger,
        run_id,
        device
    )


def fit_explainer(
        epochs,
        baseline,
        criterion,
        solver,
        train_loader,
        val_loader,
        run_manager,
        run_id,
        device
):
    run_manager.begin_run(run_id)
    for epoch in range(epochs):
        run_manager.begin_epoch()
        running_loss = 0
        baseline.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, (images, target) in enumerate(train_loader):
                images = images.to(device)
                target = target.to(device)
                solver.zero_grad()
                y_hat = baseline(images)
                train_loss = criterion(y_hat, target)
                train_loss.backward()
                solver.step()
                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_per_epoch(y_hat, target)

                running_loss += train_loss.item()
                t.set_postfix(epoch='{0}'.format(epoch), training_loss='{:05.3f}'.format(running_loss))
                t.update()

        baseline.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, (images, target) in enumerate(val_loader):
                    images = images.to(device)
                    target = target.to(device)
                    y_hat = baseline(images)
                    val_loss = criterion(y_hat, target)

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_per_epoch(y_hat, target)
                    run_manager.track_val_bb_outputs(out_class=y_hat, val_y=target)
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss))
                    t.update()

        run_manager.end_epoch(baseline, solver, multi_label=False)
        print(f"Epoch: [{epoch + 1}/{epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} (%) "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"Best_Val_AUROC: {round(run_manager.best_auroc, 4)}  "
              f"Val_Accuracy: {round(run_manager.val_accuracy, 4)} (%)  "
              f"Val_AUROC: {round(run_manager.val_auroc, 4)} (0-1) "
              f"Val_AURPC: {round(run_manager.val_aurpc, 4)} (0-1) "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)} secs")

    run_manager.end_run()
