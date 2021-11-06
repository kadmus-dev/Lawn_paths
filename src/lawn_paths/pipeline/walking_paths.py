import pytorch_lightning as pl
import torchmetrics
from lawn_paths.pipeline.functions import *
from lawn_paths.pipeline.utils import *


def BCELoss_class_weighted(weights):
    def loss(pred, target):
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        bce = - weights[1] * target * torch.log(pred) - (1 - target) * weights[0] * torch.log(1 - pred)
        return torch.mean(bce)

    return loss


class WalkingPathsDetector(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self._hparams = hparams

        self.model = object_from_dict(self._hparams["model"])
        if "resume_from_checkpoint" in self._hparams:
            corrections: Dict[str, str] = {"model.": ""}

            state_dict = state_dict_from_disk(
                file_path=self._hparams["resume_from_checkpoint"],
                rename_in_layers=corrections,
            )
            self.model.load_state_dict(state_dict)

        self.loss = BCELoss_class_weighted(weights=[1, 10])

    def forward(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(batch)

    def setup(self, stage=0):
        with open(self._hparams['train_samples']) as f:
            self.train_samples = list(json.load(f).items())

        with open(self._hparams['val_samples']) as f:
            self.val_samples = list(json.load(f).items())

        print("Len train samples = ", len(self.train_samples))
        print("Len val samples = ", len(self.val_samples))

    def predict(self, image, block_size):
        image_blocks = get_blocks(image, block_size)

        #         transform = from_dict(self._hparams["val_aug"])

        with torch.no_grad():
            result_blocks = []
            for i in range(len(image_blocks)):
                block = image_blocks[i].astype(np.float32)
                #                 block = cv2.cvtColor(image_blocks[i], cv2.COLOR_RGB2BGR).astype(np.float32)
                #                 normalized = transform(image=block)["image"]
                output = self(torch.from_numpy(block.transpose(2, 0, 1))[None, ...].type(torch.FloatTensor).cpu()).cpu()
                output = np.round(output.numpy())
                result_blocks.append(output[0, 0, :, :])
            result_blocks = np.array(result_blocks)

            img_size = image.shape[0]
            pad_width = (block_size - (img_size % block_size)) // 2
            mask_shape = (img_size + 2 * pad_width, img_size + 2 * pad_width)
            mask = np.zeros(mask_shape, dtype=np.uint8)
            height, width = mask_shape
            num = 0
            for j in range(height // block_size):
                for i in range(width // block_size):
                    up, down = j * block_size, (j + 1) * block_size
                    left, right = i * block_size, (i + 1) * block_size
                    mask[up:down, left:right] = result_blocks[num][:, :]
                    num += 1
            return mask[pad_width:img_size + pad_width, pad_width:img_size + pad_width]

    def configure_optimizers(self):

        optimizer = object_from_dict(
            self._hparams["optimizer"],
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        scheduler = object_from_dict(self._hparams["scheduler"]["lr_scheduler"], optimizer=optimizer)
        lr_dict = self._hparams["scheduler"]["lr_dict"]
        lr_dict["scheduler"] = scheduler

        self.optimizers = [optimizer]
        return self.optimizers, [lr_dict]

    def training_step(self, batch, batch_idx):
        features, gt = batch

        preds = self.forward(features).type(torch.FloatTensor).cpu()

        logs = {}

        # calculating loss
        total_loss = self.loss(preds, gt)

        logs["train_loss"] = total_loss.detach()
        logs["lr"] = self._get_current_lr()

        # calculating metrics
        precision = torchmetrics.Precision()
        recall = torchmetrics.Recall()
        f1 = torchmetrics.F1()

        logs["precision"] = precision(preds.cpu(), gt.type(torch.IntTensor).cpu()).detach()
        logs["recall"] = recall(preds.cpu(), gt.type(torch.IntTensor).cpu()).detach()
        logs["f1"] = f1(preds.cpu(), gt.type(torch.IntTensor).cpu()).detach()

        return {"loss": total_loss, "logs": logs}

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0].cpu()

    def validation_step(self, batch, batch_id):
        features, gt = batch

        preds = self.forward(features).type(torch.FloatTensor).cpu()

        logs = {"val_loss": self.loss(preds, gt)}

        # calculating metrics
        precision = torchmetrics.Precision()
        recall = torchmetrics.Recall()
        f1 = torchmetrics.F1()

        logs["val_precision"] = precision(preds.cpu(), gt.type(torch.IntTensor).cpu()).detach()
        logs["val_recall"] = recall(preds.cpu(), gt.type(torch.IntTensor).cpu()).detach()
        logs["val_f1"] = f1(preds.cpu(), gt.type(torch.IntTensor).cpu()).detach()

        return {"logs": logs}

    def training_epoch_end(self, outputs):
        logs = {"epoch": self.trainer.current_epoch,
                "train_loss": torch.stack([x['logs']['train_loss'] for x in outputs]).mean(),
                "precision": torch.stack([x['logs']['precision'] for x in outputs]).mean(),
                "recall": torch.stack([x['logs']['recall'] for x in outputs]).mean(),
                "f1": torch.stack([x['logs']['f1'] for x in outputs]).mean()}

        print(
            f'[Epoch {logs["epoch"]:3}] train_loss: {logs["train_loss"]:.4f}, f1: {logs["f1"]:.4f}, precision: {logs["precision"]:.4f}, recall: {logs["recall"]:.4f}',
            end="")

        for name, value in logs.items():
            self.log(name, value, prog_bar=True, on_epoch=True, on_step=False)

    def validation_epoch_end(self, outputs):
        logs = {"epoch": self.trainer.current_epoch,
                "val_loss": torch.stack([x['logs']['val_loss'] for x in outputs]).mean(),
                "val_precision": torch.stack([x['logs']['val_precision'] for x in outputs]).mean(),
                "val_recall": torch.stack([x['logs']['val_recall'] for x in outputs]).mean(),
                "val_f1": torch.stack([x['logs']['val_f1'] for x in outputs]).mean()}

        print(
            f'[Epoch {logs["epoch"]:3}] val_loss: {logs["val_loss"]:.4f}, val_f1: {logs["val_f1"]:.4f}, val_precision: {logs["val_precision"]:.4f}, val_recall: {logs["val_recall"]:.4f}',
            end="")

        for name, value in logs.items():
            self.log(name, value, prog_bar=True, on_epoch=True, on_step=False)
