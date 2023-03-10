import pytorch_lightning as pl
import sklearn.metrics
import torch

from Baseline.models.cem import ConceptEmbeddingModel


def construct_model(
    n_concepts,
    n_tasks,
    config,
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
    if config["architecture"] in ["ConceptEmbeddingModel", "MixtureEmbModel"]:
        model_cls = ConceptEmbeddingModel
        extra_params = {
            "emb_size": config["emb_size"],
            "shared_prob_gen": config["shared_prob_gen"],
            "sigmoidal_prob": config.get("sigmoidal_prob", False),
            "sigmoidal_embedding": config.get("sigmoidal_embedding", False),
            "intervention_idxs": intervention_idxs,
            "adversarial_intervention": adversarial_intervention,
            "training_intervention_prob": config.get(
                'training_intervention_prob',
                0.0,
            ),
            "concat_prob": config.get("concat_prob", False),
            "embeding_activation": config.get("embeding_activation", None),
        }
    elif "ConceptBottleneckModel" in config["architecture"]:
        model_cls = ConceptBottleneckModel
        extra_params = {
            "bool": config["bool"],
            "extra_dims": config["extra_dims"],
            "sigmoidal_extra_capacity": config.get(
                "sigmoidal_extra_capacity",
                True,
            ),
            "sigmoidal_prob": config.get("sigmoidal_prob", True),
            "intervention_idxs": intervention_idxs,
            "adversarial_intervention": adversarial_intervention,
            "c2y_layers": config.get("c2y_layers", []),
            "bottleneck_nonlinear": config.get("bottleneck_nonlinear", None),
            "active_intervention_values": active_intervention_values,
            "inactive_intervention_values": inactive_intervention_values,
            "x2c_model": x2c_model,
            "c2y_model": c2y_model,
        }
    else:
        raise ValueError(f'Invalid architecture "{config["architecture"]}"')

    if isinstance(config["c_extractor_arch"], str):
        if config["c_extractor_arch"] == "resnet18":
            c_extractor_arch = resnet18
        elif config["c_extractor_arch"] == "resnet34":
            c_extractor_arch = resnet34
        elif config["c_extractor_arch"] == "resnet50":
            c_extractor_arch = resnet50
        elif config["c_extractor_arch"] == "densenet121":
            c_extractor_arch = densenet121
        else:
            raise ValueError(f'Invalid model_to_use "{config["model_to_use"]}"')
    else:
        c_extractor_arch = config["c_extractor_arch"]

    # Create model
    return model_cls(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        weight_loss=(
            torch.FloatTensor(imbalance)
            if config['weight_loss'] and (imbalance is not None)
            else None
        ),
        concept_loss_weight=config['concept_loss_weight'],
        task_loss_weight=config.get('task_loss_weight', 1.0),
        normalize_loss=config['normalize_loss'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        pretrain_model=config.get('pretrain_model', False),
        c_extractor_arch=c_extractor_arch,
        optimizer=config['optimizer'],
        top_k_accuracy=config.get('top_k_accuracy'),
        **extra_params,
    )

class WrapperModule(pl.LightningModule):
    def __init__(
            self,
            model,
            n_tasks,
            model_type="x2c",
            loss=None,
            momentum=0.9,
            learning_rate=0.01,
            weight_decay=4e-05,
            optimizer="sgd",
            top_k_accuracy=2,
            binary_output=False,
            weight_loss=None,
            sigmoidal_output=False,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.binary_output = binary_output
        self.model = model
        if loss == "BCE_w_logit":
            self.loss_task = torch.nn.BCEWithLogitsLoss()
        elif loss == "Cross_Ent_loss":
            self.loss_task = torch.nn.CrossEntropyLoss()
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.top_k_accuracy = top_k_accuracy
        self.model_type = model_type
        if sigmoidal_output:
            self.sig = torch.nn.Sigmoid()
            self.acc_sig = lambda x: x
        else:
            # Then we assume the model already outputs a sigmoidal vector
            self.sig = lambda x: x
            self.acc_sig = (
                torch.nn.Sigmoid() if self.binary_output else lambda x: x
            )

    def forward(self, x):
        return self.sig(self.model(x))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, c = batch
        return self(x)

    def _run_step(self, batch, batch_idx, train=False):
        x, y, c = batch
        y_logits = self(x)
        gt = None
        if self.model_type == "x2c":
            gt = c
        elif self.model_type == "y2c":
            gt = y
        loss = self.loss_task(
            y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
            gt,
        )
        # compute accuracy
        (y_accuracy, y_auc, y_f1) = compute_accuracy(
            y_true=y,
            y_pred=self.acc_sig(y_logits),
            binary_output=self.binary_output,
        )

        result = {
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "loss": loss.detach(),
        }
        if (self.top_k_accuracy is not None) and (self.n_tasks > 2) and (
                not self.binary_output
        ):
            y_true = y.reshape(-1).cpu().detach()
            y_pred = y_logits.cpu().detach()
            labels = list(range(self.n_tasks))
            for top_k_val in self.top_k_accuracy:
                y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                    y_true,
                    y_pred,
                    k=top_k_val,
                    labels=labels,
                )
                result[f'y_top_{top_k_val}_accuracy'] = y_top_k_accuracy
        return loss, result

    def training_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=True)
        for name, val in result.items():
            self.log(name, val, prog_bar=("accuracy" in name))
        return {
            "loss": loss,
            "log": {
                "y_accuracy": result['y_accuracy'],
                "y_auc": result['y_auc'],
                "y_f1": result['y_f1'],
                "loss": result['loss'],
            },
        }

    def validation_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("val_" + name, val, prog_bar=("accuracy" in name))
        return {
            "val_" + key: val
            for key, val in result.items()
        }

    def test_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("test_" + name, val, prog_bar=True)
        return result['loss']

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }


def compute_bin_accuracy(y_pred, y_true):
    y_probs = y_pred.reshape(-1).cpu().detach()
    y_pred = y_probs > 0.5
    y_true = y_true.reshape(-1).cpu().detach()
    y_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    try:
        y_auc = sklearn.metrics.roc_auc_score(
            y_true,
            y_probs,
            multi_class='ovo',
        )
    except:
        y_auc = 0
    try:
        y_f1 = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    except:
        y_f1 = 0
    return (y_accuracy, y_auc, y_f1)


def compute_accuracy(
        y_pred,
        y_true,
        binary_output=False,
):
    if (len(y_pred.shape) < 2) or (y_pred.shape[-1] == 1) or binary_output:
        return compute_bin_accuracy(
            y_pred=y_pred,
            y_true=y_true,
        )
    y_probs = torch.nn.Softmax(dim=-1)(y_pred).cpu().detach()
    used_classes = np.unique(y_true.reshape(-1).cpu().detach())
    y_probs = y_probs[:, sorted(list(used_classes))]
    y_pred = y_pred.argmax(dim=-1).cpu().detach()
    y_true = y_true.reshape(-1).cpu().detach()
    y_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    try:
        y_auc = sklearn.metrics.roc_auc_score(
            y_true,
            y_probs,
            multi_class='ovo',
        )
    except:
        y_auc = 0.0
    y_f1 = 0.0
    return (y_accuracy, y_auc, y_f1)
