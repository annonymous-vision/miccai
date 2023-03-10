import builtins
import os
import pickle
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from Baseline.models.baseline_mimic import BaseLine_MIMIC
from Explainer.loss_F import entropy_loss
from Explainer.models.explainer import Explainer
from Logger.logger_mimic_cxr import Logger_MIMIC_CXR
from dataset.dataset_mimic_cxr import MIMICCXRDataset, Dataset_mimic_for_explainer
import MIMIC_CXR.mimic_cxr_utils as FOL_mimic


def test_explainer(args):
    random.seed(args.cur_seed)
    np.random.seed(args.cur_seed)
    torch.manual_seed(args.cur_seed)
    args.concept_names = args.landmark_names_spec + args.abnorm_obs_concepts
    if len(args.selected_obs) == 1:
        disease_folder = args.selected_obs[0]
    else:
        disease_folder = f"{args.selected_obs[0]}_{args.selected_obs[1]}"

    hidden_layers = ""
    for hl in args.hidden_nodes:
        hidden_layers += str(hl)

    root = f"lr_{args.lr}_epochs_{args.epochs}"
    args.metric = "auroc"
    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "Baseline", "Explainer", root, args.arch, disease_folder,
                               f"seed_{args.cur_seed}")
    output_path = os.path.join(args.output, args.dataset, "Baseline", "Explainer", root, args.arch, disease_folder,
                               f"seed_{args.cur_seed}")
    tb_logs_path = os.path.join(
        args.logs, args.dataset, "Baseline", "Explainer", f"{root}_{args.arch}_{disease_folder}"
    )
    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)
    print("############## Paths ################")
    print(chk_pt_path)
    print(output_path)
    print(tb_logs_path)
    print("############## Paths ###############")
    device = utils.get_device()
    print(f"Device: {device}")
    print(output_path)
    dataset_path = os.path.join(args.dataset_folder_concepts, "dataset_g")
    print(dataset_path)
    args.concept_names = pickle.load(open(os.path.join(dataset_path, f"selected_concepts_{args.metric}.pkl"), "rb"))
    print("Concepts:")
    print(args.concept_names)
    print(f"Length of concepts: {len(args.concept_names)}")
    pickle.dump(args, open(os.path.join(output_path, "test_explainer_configs.pkl"), "wb"))

    start = time.time()
    train_dataset = Dataset_mimic_for_explainer(
        iteration=1,
        mode="train",
        expert="baseline",
        dataset_path=dataset_path,
        metric="auroc"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_dataset = Dataset_mimic_for_explainer(
        iteration=1,
        mode="val",
        expert="baseline",
        dataset_path=dataset_path,
        metric="auroc"
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    test_dataset = Dataset_mimic_for_explainer(
        iteration=1,
        mode="test",
        expert="baseline",
        dataset_path=dataset_path,
        metric="auroc"
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    print(f"Train Dataset: {len(train_dataset)}")
    print(f"Val Dataset: {len(val_dataset)}")
    print(f"Test Dataset: {len(test_dataset)}")
    done = time.time()
    elapsed = done - start
    print("Time to load the dataset: " + str(elapsed) + " secs")
    model = Explainer(
        n_concepts=len(args.concept_names), n_classes=len(args.labels), explainer_hidden=args.hidden_nodes,
        conceptizator=args.conceptizator, temperature=args.temperature_lens,
    ).to(device)
    cur_glt_chkpt = os.path.join(chk_pt_path, "best_model.pth.tar")
    model.load_state_dict(torch.load(cur_glt_chkpt)["state_dict"])
    model.eval()

    predict(args, model, train_loader, output_path, mode="train")
    predict(args, model, val_loader, output_path, mode="val")
    predict(args, model, test_loader, output_path, mode="test")


def predict(args, model, loader, output_path, mode):
    out_put_class = torch.FloatTensor().cuda()
    out_put_target = torch.FloatTensor().cuda()
    proba_concept = torch.FloatTensor().cuda()
    attributes_gt = torch.FloatTensor().cuda()
    tensor_concept_mask = torch.FloatTensor().cuda()
    tensor_alpha = torch.FloatTensor().cuda()
    tensor_alpha_norm = torch.FloatTensor().cuda()

    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for batch_id, data in enumerate(loader):
                (
                    logits_concept_x,
                    proba_concept_x,
                    attributes_gt_x,
                    y,
                    y_one_hot,
                    concepts
                ) = data
                if torch.cuda.is_available():
                    proba_concept_x = proba_concept_x.cuda(args.gpu, non_blocking=True)
                    attributes_gt_x = attributes_gt_x.cuda(args.gpu, non_blocking=True)
                    y = y.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                out_class = model(proba_concept_x)
                out_put_class = torch.cat((out_put_class, out_class), dim=0)
                out_put_target = torch.cat((out_put_target, y), dim=0)
                proba_concept = torch.cat((proba_concept, proba_concept_x), dim=0)
                attributes_gt = torch.cat((attributes_gt, attributes_gt_x), dim=0)

                tensor_concept_mask = model.model[0].concept_mask
                tensor_alpha = model.model[0].alpha
                tensor_alpha_norm = model.model[0].alpha_norm

                t.set_postfix(batch_id="{0}".format(batch_id))
                t.update()

    out_put_class_pred = out_put_class.cpu()
    out_put_target = out_put_target.cpu()
    proba_concept = proba_concept.cpu()
    attributes_gt = attributes_gt.cpu()
    tensor_alpha_norm = tensor_alpha_norm.cpu()
    tensor_concept_mask = tensor_concept_mask.cpu()
    tensor_alpha = tensor_alpha.cpu()

    print(f"out_put_class_pred size: {out_put_class_pred.size()}")
    print(f"out_put_target size: {out_put_target.size()}")
    print(f"proba_concept size: {proba_concept.size()}")
    print(f"attributes_gt size: {attributes_gt.size()}")
    print(f"tensor_alpha_norm size: {tensor_alpha_norm.size()}")

    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_out_put_class_pred.pt"), tensor_to_save=out_put_class_pred
    )
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_out_put_target.pt"), tensor_to_save=out_put_target)
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_proba_concept.pt"), tensor_to_save=proba_concept)
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_attributes_gt.pt"), tensor_to_save=attributes_gt)

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

    print("Creating Explanations")
    x_to_bool = 0.5
    device, configs = FOL_mimic.setup(output_path)
    df_master_sel = FOL_mimic.load_master_csv(configs, mode)
    percentile_selection = 99
    iteration = 1
    test_tensor_concepts_bool = (proba_concept.cpu() > x_to_bool).to(torch.float)
    _feature_names = [f"feature{j:010}" for j in range(test_tensor_concepts_bool.size(1))]
    results_arr = []
    ii = 0
    for _idx in range(out_put_class_pred.size(0)):
        ii += 1
        print(f"===>> {ii} <<===")
        results = FOL_mimic.compute_explanations_per_sample(
            iteration,
            _idx,
            df_master_sel,
            _feature_names,
            out_put_class_pred,
            out_put_class_pred,
            out_put_target,
            test_tensor_concepts_bool,
            tensor_alpha_norm,
            percentile_selection,
            args.concept_names,
            model,
            proba_concept,
            attributes_gt,
            device,
            model_type="baseline"
        )
        results_arr.append(results)
        print(
            f" {[results['idx']]}, predicted: {configs.labels[results['g_pred']]}, target: {configs.labels[results['ground_truth']]}"
        )
        print(f" {configs.labels[results['g_pred']]} <=> {results['actual_explanations']}")

    pickle.dump(results_arr, open(os.path.join(output_path, f"{mode}_FOL_results_baseline_cbm.pkl"), "wb"))
    test_results_df = pd.DataFrame.from_dict(results_arr, orient='columns')
    test_results_df.to_csv(os.path.join(output_path, f"{mode}_FOL_results_baseline_cbm.csv"))


def test_backbone(args):
    print("###############################################")
    print("Testing backbone")
    args.N_landmarks_spec = len(args.landmark_names_spec)
    args.N_selected_obs = len(args.selected_obs)
    args.N_abnorm_obs_concepts = len(args.abnorm_obs_concepts)
    args.N_labels = len(args.labels)

    if len(args.selected_obs) == 1:
        disease_folder = args.selected_obs[0]
    else:
        disease_folder = f"{args.selected_obs[0]}_{args.selected_obs[1]}"

    root = f"lr_{args.lr}_epochs_{args.epochs}"

    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "Baseline", "Backbone", root, args.arch, disease_folder)
    output_path = os.path.join(args.output, args.dataset, "Baseline", "Backbone", root, args.arch, disease_folder)
    tb_logs_path = os.path.join(args.logs, args.dataset, "Baseline", "Backbone", f"{root}_{args.arch}_{disease_folder}")
    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)
    print("############## Paths ################")
    print(chk_pt_path)
    print(output_path)
    print(tb_logs_path)
    print("############## Paths ###############")
    device = utils.get_device()
    print(f"Device: {device}")
    print(output_path)

    pickle.dump(args, open(os.path.join(output_path, "MIMIC_test_configs.pkl"), "wb"))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        """
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        """

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = args.ngpus_per_node
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker_test, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker_test(args.gpu, ngpus_per_node, args, chk_pt_path, output_path)


def main_worker_test(gpu, ngpus_per_node, args, chk_pt_path, output_path):
    args.gpu = gpu
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank
        )

    # create model
    print("=> Creating model '{}'".format(args.arch))
    baseline = BaseLine_MIMIC(args)
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            baseline.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model_bb = torch.nn.parallel.DistributedDataParallel(baseline, device_ids=[args.gpu])
        else:
            baseline.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            baseline = torch.nn.parallel.DistributedDataParallel(baseline)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        baseline = baseline.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            baseline.features = torch.nn.DataParallel(baseline.features)
            baseline.cuda()
        else:
            baseline = torch.nn.DataParallel(baseline).cuda()

    # optionally resume from a checkpoint
    best_auroc = 0
    cudnn.benchmark = True
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    arr_rad_graph_sids = np.load(args.radgraph_sids_npy_file)  # N_sids
    arr_rad_graph_adj = np.load(args.radgraph_adj_mtx_npy_file)  # N_sids * 51 * 75

    start = time.time()
    test_dataset = MIMICCXRDataset(
        args=args,
        radgraph_sids=arr_rad_graph_sids,
        radgraph_adj_mtx=arr_rad_graph_adj,
        mode='test',
        transform=transforms.Compose([
            transforms.Resize(args.resize),
            transforms.CenterCrop(args.resize),
            transforms.ToTensor(),  # convert pixel value to [0, 1]
            normalize
        ]),
        model_type="t"
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=args.workers, shuffle=False, pin_memory=True,
        drop_last=True
    )

    done = time.time()
    elapsed = done - start
    print("Time to load the dataset: " + str(elapsed) + " secs")

    start = time.time()
    model_chk_pt = torch.load(os.path.join(chk_pt_path, args.checkpoint_bb))
    if "state_dict" in model_chk_pt:
        baseline.load_state_dict(model_chk_pt['state_dict'])
    else:
        baseline.load_state_dict(model_chk_pt)

    done = time.time()
    elapsed = done - start
    print("Time to load the BB: " + str(elapsed) + " secs")
    args.save_concepts = True
    if args.save_concepts:
        start = time.time()
        train_dataset = MIMICCXRDataset(
            args=args,
            radgraph_sids=arr_rad_graph_sids,
            radgraph_adj_mtx=arr_rad_graph_adj,
            mode="train",
            transform=transforms.Compose(
                [
                    transforms.Resize(args.resize),
                    # resize smaller edge to args.resize and the aspect ratio the same for the longer edge
                    transforms.CenterCrop(args.resize),
                    # transforms.RandomRotation(args.degree),
                    # transforms.RandomCrop(args.crop),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # convert pixel value to [0, 1]
                    normalize,
                ]
            ),
            model_type="t",
            network_type=args.network_type,
            feature_path=args.feature_path
        )

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, num_workers=args.workers, shuffle=False,
            pin_memory=True, sampler=train_sampler, drop_last=True
        )

        val_dataset = MIMICCXRDataset(
            args=args,
            radgraph_sids=arr_rad_graph_sids,
            radgraph_adj_mtx=arr_rad_graph_adj,
            mode='valid',
            transform=transforms.Compose([
                transforms.Resize(args.resize),
                transforms.CenterCrop(args.resize),
                transforms.ToTensor(),  # convert pixel value to [0, 1]
                normalize
            ]),
            model_type="t",
            network_type=args.network_type,
            feature_path=args.feature_path
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=args.workers, shuffle=False, pin_memory=True,
            drop_last=True
        )

        done = time.time()
        elapsed = done - start
        print("Time to load the dataset: " + str(elapsed) + " secs")
        output_path_t_dataset_g = os.path.join(output_path, "dataset_g")
        os.makedirs(output_path_t_dataset_g, exist_ok=True)

        print("=============>> Saving concepts for test set")
        save_concepts(
            args,
            test_loader,
            baseline,
            output_path_t_dataset_g,
            mode="test"
        )
        print("=============>> Saving concepts for training set")
        save_concepts(
            args,
            train_loader,
            baseline,
            output_path_t_dataset_g,
            mode="train"
        )
        print("=============>> Saving concepts for val set")
        save_concepts(
            args,
            val_loader,
            baseline,
            output_path_t_dataset_g,
            mode="val"
        )

    print("Validation tensors are saving")
    validate(args, baseline, test_loader, output_path)


def save_concepts(
        args,
        loader,
        baseline,
        output_path,
        mode
):
    feature_path = os.path.join(output_path, f"{mode}_features")
    transformed_img_path = os.path.join(output_path, f"{mode}_transformed_images")
    raw_img_path = os.path.join(output_path, f"{mode}_raw_images")
    os.makedirs(feature_path, exist_ok=True)
    os.makedirs(transformed_img_path, exist_ok=True)
    os.makedirs(raw_img_path, exist_ok=True)

    # features_phi = torch.FloatTensor()
    logits_concepts_x = torch.FloatTensor().cuda()
    proba_concepts_x = torch.FloatTensor().cuda()
    class_labels = torch.FloatTensor().cuda()
    attributes = torch.FloatTensor().cuda()
    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for batch_id, data in enumerate(loader):
                (
                    dicom_id,
                    image,
                    vit_features,
                    densenet_features,
                    raw_image,
                    adj_mtx,
                    chexpert_label,
                    _,
                    landmark_spec_label,
                    landmarks_spec_inverse_weight,
                    landmark_spec_label_pnu,
                    selected_obs_label_gt,
                    selected_obs_inverse_weight,
                    selected_obs_label_pnu,
                    full_obs_label_gt,
                    full_obs_inverse_weight,
                    full_obs_label_pnu,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = data
                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True)

                if torch.cuda.is_available():
                    selected_obs_label_gt = selected_obs_label_gt.cuda(args.gpu, non_blocking=True)
                    landmark_spec_label = landmark_spec_label.cuda(args.gpu, non_blocking=True)
                    full_obs_label_gt = full_obs_label_gt.cuda(args.gpu, non_blocking=True)

                gt = torch.cat((landmark_spec_label, full_obs_label_gt), dim=1)
                logits_concepts = baseline(image)
                y_hat = torch.sigmoid(logits_concepts)

                utils.save_tensor(
                    path=os.path.join(transformed_img_path, f"transformed_img_{batch_id}.pth.tar"),
                    tensor_to_save=image.cpu()
                )
                utils.save_tensor(
                    path=os.path.join(raw_img_path, f"raw_img_{batch_id}.pth.tar"),
                    tensor_to_save=raw_image.cpu()
                )

                # features_phi = torch.cat((features_phi, features.cpu()), dim=0)
                logits_concepts_x = torch.cat((logits_concepts_x, logits_concepts), dim=0)
                proba_concepts_x = torch.cat((proba_concepts_x, y_hat), dim=0)
                class_labels = torch.cat((class_labels, selected_obs_label_gt), dim=0)
                attributes = torch.cat((attributes, gt), dim=0)

                t.set_postfix(batch_id='{0}'.format(batch_id))
                t.update()

    logits_concepts_x = logits_concepts_x.cpu()
    proba_concepts_x = proba_concepts_x.cpu()
    class_labels = class_labels.cpu()
    attributes = attributes.cpu()

    # print(f"====> Saved features phi size: {features_phi.size()}")
    print(f"====> Saved logits concepts_x size: {logits_concepts_x.size()}")
    print(f"====> Saved proba concepts_x size: {proba_concepts_x.size()}")
    print(f"====> Saved class_labels size: {class_labels.size()}")
    print(f"====> Saved attributes size: {attributes.size()}")

    # utils.save_tensor(path=os.path.join(output_path, f"{mode}_features_phi.pt"), tensor_to_save=features_phi)
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_logits_concepts.pt"), tensor_to_save=logits_concepts_x)
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_proba_concepts.pt"), tensor_to_save=proba_concepts_x)
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_class_labels.pt"), tensor_to_save=class_labels)
    utils.save_tensor(path=os.path.join(output_path, f"{mode}_attributes.pt"), tensor_to_save=attributes)

    # print(f"====> features phi saved at {os.path.join(output_path, f'{mode}_features_phi.pt')}")
    print(f"====> Logits Concepts saved at {os.path.join(output_path, f'{mode}_logits_concepts.pt')}")
    print(f"====> Proba Concepts saved at {os.path.join(output_path, f'{mode}_proba_concepts.pt')}")
    print(f"====> Class labels saved at {os.path.join(output_path, f'{mode}_class_labels.pt')}")
    print(f"====> Attributes labels saved at {os.path.join(output_path, f'{mode}_attributes.pt')}")


def validate(args, baseline, loader, output_path):
    baseline.eval()
    concept_names = args.landmark_names_spec + args.abnorm_obs_concepts

    out_prob_arr_bb = []
    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict = torch.FloatTensor().cuda()
    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for batch_id, data in enumerate(loader):
                (
                    dicom_id,
                    image,
                    vit_features,
                    densenet_features,
                    raw_image,
                    adj_mtx,
                    chexpert_label,
                    _,
                    landmark_spec_label,
                    landmarks_spec_inverse_weight,
                    landmark_spec_label_pnu,
                    selected_obs_label_gt,
                    selected_obs_inverse_weight,
                    selected_obs_label_pnu,
                    full_obs_label_gt,
                    full_obs_inverse_weight,
                    full_obs_label_pnu,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = data

                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True)

                if torch.cuda.is_available():
                    landmark_spec_label = landmark_spec_label.cuda(args.gpu, non_blocking=True)
                    full_obs_label_gt = full_obs_label_gt.cuda(args.gpu, non_blocking=True)

                gt = torch.cat((landmark_spec_label, full_obs_label_gt), dim=1)
                logits_concepts = baseline(image)
                y_hat = torch.sigmoid(logits_concepts)

                out_put_predict = torch.cat((out_put_predict, y_hat), dim=0)
                out_put_GT = torch.cat((out_put_GT, gt), dim=0)
                t.set_postfix(batch_id='{0}'.format(batch_id))
                t.update()

    out_put_GT_np = out_put_GT.cpu().numpy()
    out_put_predict_np = out_put_predict.cpu().numpy()
    y_pred = np.where(out_put_predict_np > 0.5, 1, 0)

    # cls_report = {}
    # for i, concept_name in enumerate(concept_names):
    #     cls_report[concept_name] = {}
    # for i, concept_name in enumerate(concept_names):
    #     cls_report[concept_name]["accuracy"] = metrics.accuracy_score(y_pred=y_pred[i], y_true=out_put_GT_np[i])
    #     cls_report[concept_name]["precision"] = metrics.precision_score(y_pred=y_pred[i], y_true=out_put_GT_np[i])
    #     cls_report[concept_name]["recall"] = metrics.recall_score(y_pred=y_pred[i], y_true=out_put_GT_np[i])
    #     cls_report[concept_name]["f1"] = metrics.f1_score(y_pred=y_pred[i], y_true=out_put_GT_np[i])
    #
    # cls_report["accuracy_overall"] = (y_pred == out_put_GT_np).sum() / (out_put_GT_np.shape[0] * out_put_GT_np.shape[1])
    # for i, concept_name in enumerate(concept_names):
    #     print(f"{concept_name}: {cls_report[concept_name]}")
    #
    # print(f"Overall Accuracy: {cls_report['accuracy_overall']}")

    out_AUROC = utils.compute_AUROC(
        out_put_GT,
        out_put_predict,
        len(concept_names)
    )

    auroc_mean = np.array(out_AUROC).mean()
    print("<<< Model Test Results: AUROC >>>")
    print("MEAN", ": {:.4f}".format(auroc_mean))

    for i in range(0, len(out_AUROC)):
        print(concept_names[i], ': {:.4f}'.format(out_AUROC[i]))
    print("------------------------")

    # utils.dump_in_pickle(output_path=output_path, file_name="cls_report.pkl", stats_to_dump=cls_report)
    utils.dump_in_pickle(output_path=output_path, file_name="AUC_ROC.pkl", stats_to_dump=out_AUROC)

    # print(f"Classification report is saved at {output_path}/cls_report.pkl")
    print(f"AUC-ROC report is saved at {output_path}/AUC_ROC.pkl")


def train_backbone(args):
    print("###############################################")
    args.N_landmarks_spec = len(args.landmark_names_spec)
    args.N_selected_obs = len(args.selected_obs)
    args.N_abnorm_obs_concepts = len(args.abnorm_obs_concepts)
    args.N_labels = len(args.labels)

    if len(args.selected_obs) == 1:
        disease_folder = args.selected_obs[0]
    else:
        disease_folder = f"{args.selected_obs[0]}_{args.selected_obs[1]}"

    root = f"lr_{args.lr}_epochs_{args.epochs}"

    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "Baseline", "Backbone", root, args.arch, disease_folder)
    output_path = os.path.join(args.output, args.dataset, "Baseline", "Backbone", root, args.arch, disease_folder)
    tb_logs_path = os.path.join(args.logs, args.dataset, "Baseline", "Backbone", f"{root}_{args.arch}_{disease_folder}")
    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)

    device = utils.get_device()
    print(f"Device: {device}")
    print(output_path)
    pickle.dump(args, open(os.path.join(output_path, "MIMIC_train_configs.pkl"), "wb"))
    print(os.path.join(output_path, "MIMIC_train_configs.pkl"))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        """
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        """

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = args.ngpus_per_node
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, chk_pt_path, output_path, tb_logs_path)


def main_worker(gpu, ngpus_per_node, args, chk_pt_path, output_path, tb_logs_path):
    args.gpu = gpu
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for testing".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank
        )

    # create model
    print("=> Creating model '{}'".format(args.arch))

    baseline = BaseLine_MIMIC(args)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            baseline.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            baseline = torch.nn.parallel.DistributedDataParallel(baseline, device_ids=[args.gpu])
        else:
            baseline.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            baseline = torch.nn.parallel.DistributedDataParallel(baseline)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        baseline = baseline.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            baseline.features = torch.nn.DataParallel(baseline.features)
            baseline.cuda()
        else:
            model_bb = torch.nn.DataParallel(baseline).cuda()

    optimizer = torch.optim.SGD(
        [
            {'params': list(baseline.linear.parameters()), 'lr': args.lr,
             'weight_decay': args.weight_decay, 'momentum': args.momentum
             }
        ])

    # optionally resume from a checkpoint
    best_auroc = 0
    if args.resume:
        # TODO: this needs to be changed as per GLT
        ckpt_path = os.path.join(chk_pt_path_t, args.resume)
        if os.path.isfile(ckpt_path):
            config_path = os.path.join(output_path, 'MIMIC_train_configs.pkl')
            args = pickle.load(open(config_path, "rb"))
            args.distributed = False
            baseline = BaseLine_MIMIC(args)
            checkpoint = torch.load(ckpt_path)
            args.start_epoch = checkpoint['epoch']
            best_auroc = checkpoint['best_auroc']
            baseline.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(ckpt_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    arr_rad_graph_sids = np.load(args.radgraph_sids_npy_file)  # N_sids
    arr_rad_graph_adj = np.load(args.radgraph_adj_mtx_npy_file)  # N_sids * 51 * 75

    start = time.time()
    train_dataset = MIMICCXRDataset(
        args=args,
        radgraph_sids=arr_rad_graph_sids,
        radgraph_adj_mtx=arr_rad_graph_adj,
        mode="train",
        transform=transforms.Compose(
            [
                transforms.Resize(args.resize),
                # resize smaller edge to args.resize and the aspect ratio the same for the longer edge
                transforms.CenterCrop(args.resize),
                # transforms.RandomRotation(args.degree),
                # transforms.RandomCrop(args.crop),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # convert pixel value to [0, 1]
                normalize,
            ]
        ),
        model_type="t",
        network_type=args.network_type,
        feature_path=args.feature_path
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True
    )

    val_dataset = MIMICCXRDataset(
        args=args,
        radgraph_sids=arr_rad_graph_sids,
        radgraph_adj_mtx=arr_rad_graph_adj,
        mode='test',
        transform=transforms.Compose([
            transforms.Resize(args.resize),
            transforms.CenterCrop(args.resize),
            transforms.ToTensor(),  # convert pixel value to [0, 1]
            normalize
        ]),
        model_type="t",
        network_type=args.network_type,
        feature_path=args.feature_path
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
        drop_last=True
    )

    done = time.time()
    elapsed = done - start
    print("Time to load the dataset: " + str(elapsed) + " secs")

    final_parameters = OrderedDict(
        arch=[args.arch],
        dataset=[args.dataset],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')]
    )

    print("########### Paths ###########")
    print(chk_pt_path)
    print(tb_logs_path)
    print(output_path)
    print("########### Paths ###########")
    run_id = utils.get_runs(final_parameters)[0]
    run_manager = Logger_MIMIC_CXR(
        1, best_auroc, args.start_epoch, chk_pt_path, tb_logs_path, output_path, train_loader, val_loader,
        args.N_landmarks_spec + args.N_abnorm_obs_concepts, model_type="t"
    )

    start = time.time()
    done = time.time()
    elapsed = done - start
    print("Time to load the BB: " + str(elapsed) + " secs")
    fit(args, baseline, optimizer, train_loader, val_loader, train_sampler, run_manager, run_id)


def fit(args, baseline, optimizer, train_loader, val_loader, train_sampler, run_manager, run_id):
    run_manager.begin_run(run_id)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)
        run_manager.begin_epoch()
        # switch to train mode
        baseline.train()
        with tqdm(total=len(train_loader)) as t:
            for i, data in enumerate(train_loader):
                (
                    dicom_id,
                    image,
                    vit_features,
                    densenet_features,
                    _,
                    adj_mtx,
                    chexpert_label,
                    _,
                    landmark_spec_label,
                    landmarks_spec_inverse_weight,
                    landmark_spec_label_pnu,
                    selected_obs_label_gt,
                    selected_obs_inverse_weight,
                    selected_obs_label_pnu,
                    full_obs_label_gt,
                    full_obs_inverse_weight,
                    full_obs_label_pnu,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = data

                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True)

                if torch.cuda.is_available():
                    landmark_spec_label = landmark_spec_label.cuda(args.gpu, non_blocking=True)
                    densenet_features = densenet_features.cuda(args.gpu, non_blocking=True)
                    full_obs_label_gt = full_obs_label_gt.cuda(args.gpu, non_blocking=True)
                    landmarks_spec_inverse_weight = landmarks_spec_inverse_weight.cuda(args.gpu, non_blocking=True)
                    full_obs_inverse_weight = full_obs_inverse_weight.cuda(args.gpu, non_blocking=True)

                gt = torch.cat((landmark_spec_label, full_obs_label_gt), dim=1)
                weights = torch.cat((landmarks_spec_inverse_weight, full_obs_inverse_weight), dim=1)

                logits_concepts = baseline(image)
                train_loss = compute_loss(args, logits_concepts, gt, weights)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_multilabel_per_epoch(torch.sigmoid(logits_concepts), gt)
                t.set_postfix(
                    epoch='{0}'.format(epoch),
                    training_loss='{:05.3f}'.format(run_manager.epoch_train_loss))
                t.update()

        baseline.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for i, data in enumerate(val_loader):
                    (
                        dicom_id,
                        image,
                        vit_features,
                        densenet_features,
                        _,
                        adj_mtx,
                        chexpert_label,
                        _,
                        landmark_spec_label,
                        landmarks_spec_inverse_weight,
                        landmark_spec_label_pnu,
                        selected_obs_label_gt,
                        selected_obs_inverse_weight,
                        selected_obs_label_pnu,
                        full_obs_label_gt,
                        full_obs_inverse_weight,
                        full_obs_label_pnu,
                        _,
                        _,
                        _,
                        _,
                        _,
                    ) = data

                    if args.gpu is not None:
                        image = image.cuda(args.gpu, non_blocking=True)

                    if torch.cuda.is_available():
                        landmark_spec_label = landmark_spec_label.cuda(args.gpu, non_blocking=True)
                        densenet_features = densenet_features.cuda(args.gpu, non_blocking=True)
                        full_obs_label_gt = full_obs_label_gt.cuda(args.gpu, non_blocking=True)
                        landmarks_spec_inverse_weight = landmarks_spec_inverse_weight.cuda(args.gpu, non_blocking=True)
                        full_obs_inverse_weight = full_obs_inverse_weight.cuda(args.gpu, non_blocking=True)

                    gt = torch.cat((landmark_spec_label, full_obs_label_gt), dim=1)
                    weights = torch.cat((landmarks_spec_inverse_weight, full_obs_inverse_weight), dim=1)

                    logits_concepts = baseline(image)
                    val_loss = compute_loss(args, logits_concepts, gt, weights)

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_multilabel_per_epoch(
                        torch.sigmoid(logits_concepts), gt
                    )
                    run_manager.track_val_bb_outputs(
                        out_class=torch.sigmoid(logits_concepts),
                        val_y=gt
                    )
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss)
                    )
                    t.update()

        run_manager.end_epoch(baseline, optimizer, multi_label=True)

        print(f"Epoch: [{epoch + 1}/{args.epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} (%) "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"Best_auroc: {round(run_manager.best_auroc, 4)} (0-1) "
              f"Val_Accuracy: {round(run_manager.val_accuracy, 4)} (%)  "
              f"Val_AUROC: {round(run_manager.val_auroc, 4)} (0-1) "
              f"Val_AURPC: {round(run_manager.val_aurpc, 4)} (0-1) "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)} secs")
    run_manager.end_run()


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 6 epochs"""
    lr = args.lr * (0.33 ** (epoch // 12))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_loss(args, logits, y, weights):
    if args.loss1 == 'BCE':
        loss1 = F.binary_cross_entropy(torch.sigmoid(logits), y, reduction='mean')
    elif args.loss1 == 'BCE_W':
        loss1 = F.binary_cross_entropy(torch.sigmoid(logits), y, weight=weights, reduction='mean')
    else:
        raise Exception('Invalid loss 1 type.')

    return loss1


def train_explainer(args):
    random.seed(args.cur_seed)
    np.random.seed(args.cur_seed)
    torch.manual_seed(args.cur_seed)
    args.concept_names = args.landmark_names_spec + args.abnorm_obs_concepts
    if len(args.selected_obs) == 1:
        disease_folder = args.selected_obs[0]
    else:
        disease_folder = f"{args.selected_obs[0]}_{args.selected_obs[1]}"

    hidden_layers = ""
    for hl in args.hidden_nodes:
        hidden_layers += str(hl)

    root = f"lr_{args.lr}_epochs_{args.epochs}"
    args.metric = "auroc"
    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "Baseline", "Explainer", root, args.arch, disease_folder,
                               f"seed_{args.cur_seed}")
    output_path = os.path.join(args.output, args.dataset, "Baseline", "Explainer", root, args.arch, disease_folder,
                               f"seed_{args.cur_seed}")
    tb_logs_path = os.path.join(
        args.logs, args.dataset, "Baseline", "Explainer", f"{root}_{args.arch}_{disease_folder}"
    )
    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)
    print("############## Paths ################")
    print(chk_pt_path)
    print(output_path)
    print(tb_logs_path)
    print("############## Paths ###############")
    device = utils.get_device()
    print(f"Device: {device}")
    print(output_path)

    pickle.dump(args, open(os.path.join(output_path, "MIMIC_test_configs.pkl"), "wb"))

    dataset_path = os.path.join(args.dataset_folder_concepts, "dataset_g")
    print(dataset_path)
    start = time.time()
    train_dataset = Dataset_mimic_for_explainer(
        iteration=1,
        mode="train",
        expert="baseline",
        dataset_path=dataset_path,
        metric="auroc"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_dataset = Dataset_mimic_for_explainer(
        iteration=1,
        mode="test",
        expert="baseline",
        dataset_path=dataset_path,
        metric="auroc"
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    args.concept_names = pickle.load(open(os.path.join(dataset_path, f"selected_concepts_{args.metric}.pkl"), "rb"))
    print("Concepts:")
    print(args.concept_names)
    print(f"Length of concepts: {len(args.concept_names)}")
    print(f"Train Dataset: {len(train_dataset)}")
    print(f"Val Dataset: {len(val_dataset)}")
    done = time.time()
    elapsed = done - start
    print("Time to load the dataset: " + str(elapsed) + " secs")

    model = Explainer(
        n_concepts=len(args.concept_names), n_classes=len(args.labels), explainer_hidden=args.hidden_nodes,
        conceptizator=args.conceptizator, temperature=args.temperature_lens,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_auroc = 0
    n_class = 2
    logger = Logger_MIMIC_CXR(
        1, best_auroc, args.start_epoch, chk_pt_path, tb_logs_path, output_path, train_loader, val_loader,
        n_class, model_type="g", device=device
    )

    fit_baseline(
        args,
        model,
        optimizer,
        train_loader,
        val_loader,
        criterion,
        logger,
        args.lambda_lens,
        os.path.join(root, f"baseline"),
        device
    )


def fit_baseline(
        args,
        model,
        optimizer,
        train_loader,
        val_loader,
        criterion,
        logger,
        lambda_lens,
        run_id,
        device
):
    logger.begin_run(run_id)
    for epoch in range(args.epochs):
        logger.begin_epoch()
        model.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, data in enumerate(train_loader):
                (
                    train_logits_concept_x,
                    train_proba_concept_x,
                    _,
                    train_y,
                    y_one_hot,
                    concepts
                ) = data
                if torch.cuda.is_available():
                    train_proba_concept_x = train_proba_concept_x.cuda(args.gpu, non_blocking=True)
                    train_y = train_y.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                y_hat = model(train_proba_concept_x)
                entropy_loss_elens = entropy_loss(model)
                train_loss = criterion(y_hat, train_y) + lambda_lens * entropy_loss_elens
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                logger.track_train_loss(train_loss.item())
                logger.track_total_train_correct_per_epoch(y_hat, train_y)
                t.set_postfix(
                    epoch='{0}'.format(epoch),
                    training_loss='{:05.3f}'.format(logger.epoch_train_loss))
                t.update()

        model.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data in enumerate(val_loader):
                    (
                        _,
                        valid_proba_concept_x,
                        _,
                        valid_y,
                        y_one_hot,
                        concepts
                    ) = data
                    if torch.cuda.is_available():
                        valid_proba_concept_x = valid_proba_concept_x.cuda(args.gpu, non_blocking=True)
                        valid_y = valid_y.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                    y_hat = model(valid_proba_concept_x)
                    entropy_loss_elens = entropy_loss(model)
                    val_loss = criterion(y_hat, valid_y) + lambda_lens * entropy_loss_elens

                    logger.track_val_loss(val_loss.item())
                    logger.track_total_val_correct_per_epoch(y_hat, valid_y)
                    logger.track_val_bb_outputs(out_class=y_hat, val_y=valid_y)
                    t.set_postfix(
                        epoch='{0}'.format(epoch + 1),
                        validation_loss='{:05.3f}'.format(logger.epoch_val_loss))
                    t.update()

        logger.end_epoch(model, optimizer, multi_label=False)

        print(f"Epoch: [{epoch + 1}/{args.epochs}] "
              f"Train_loss: {round(logger.get_final_train_loss(), 4)} "
              f"Train_Accuracy: {round(logger.get_final_train_accuracy(), 4)} (%) "
              f"Val_loss: {round(logger.get_final_val_loss(), 4)} "
              f"Best_Val_AUROC: {round(logger.best_auroc, 4)}  "
              f"Val_Accuracy: {round(logger.val_accuracy, 4)} (%)  "
              f"Val_AUROC: {round(logger.val_auroc, 4)} (0-1) "
              f"Val_AURPC: {round(logger.val_aurpc, 4)} (0-1) "
              f"Epoch_Duration: {round(logger.get_epoch_duration(), 4)} secs")

    logger.end_run()
