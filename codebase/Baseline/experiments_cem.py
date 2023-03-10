import torch
from pytorch_lightning import seed_everything

from Baseline.models.baseline_cub import BaseLine_CUB
from Baseline.models.cem import ConceptEmbeddingModel


def train_sequential_model(
        args,
):
    args.x2c_loss = "BCE_w_logit"
    seed_everything(args.seed)
    n_concepts = len(args.concept_names)
    n_tasks = len(args.labels)
    print(f"[Training sequential concept extractor]")
    _, seq_c2y_model = construct_sequential_models(
        n_concepts,
        n_tasks,
        args,
    )


def construct_model(
        n_concepts,
        n_tasks,
        args,
        c2y_model=None,
        x2c_model=None,
        imbalance=None,
        concept_edge_list=None,
        concept_edge_weights=None,
        intervention_idxs=None,
        adversarial_intervention=False,
        active_intervention_values=None,
        inactive_intervention_values=None,
):
    model_cls = ConceptEmbeddingModel
    extra_params = {
        "emb_size": args.emb_size,
        "shared_prob_gen": args.shared_prob_gen,
        "sigmoidal_prob": args.sigmoidal_prob,
        "sigmoidal_embedding": args.sigmoidal_embedding,
        "intervention_idxs": intervention_idxs,
        "adversarial_intervention": adversarial_intervention,
        "training_intervention_prob": args.training_intervention_prob,
        "concat_prob": args.concat_prob,
        "embeding_activation": args.embeding_activation
    }

    c_extractor_arch = BaseLine_CUB

    # Create model
    return model_cls(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        weight_loss=(
            torch.FloatTensor(imbalance)
            if args.weight_loss and (imbalance is not None)
            else None
        ),
        concept_loss_weight=args.concept_loss_weight,
        task_loss_weight=1.0,
        normalize_loss=args.normalize_loss,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pretrain_model=config.get('pretrain_model', False),
        c_extractor_arch=c_extractor_arch,
        optimizer=config['optimizer'],
        top_k_accuracy=config.get('top_k_accuracy'),
        **extra_params,
    )
