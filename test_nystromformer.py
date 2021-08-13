from nystrom_attention import Nystromformer
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from clip_gen.positional_encoding import positional_encoding


class NystromformerLM(pl.LightningModule):
    def __init__(self, max_tokens, **kwargs):
        super().__init__()
        self.model = Nystromformer(**kwargs)
        self.register_buffer(
            "positional",
            positional_encoding(kwargs["dim"], max_tokens),
            persistent=False,
        )

    def training_step(self, batch, batch_idx):
        return self.process_batch_and_compute_loss(batch)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch_and_compute_loss(batch)
        self.log("test/loss", loss)
        return loss

    def process_batch_and_compute_loss(self, batch):
        xs, y_targets = batch
        xs.requires_grad_()
        y_targets.requires_grad_()

        positional = self.positional[: xs.shape[1], :]

        ys = self.model(xs + positional)

        loss = F.mse_loss(ys, y_targets)
        return loss

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def test_constant_nystromformer():
    "Check we can learn a constant function."

    d_model = 256
    layers = 2
    toks = 512

    copies = 8192

    train_data = torch.rand(copies, toks, d_model) * 2 - 1
    test_data = torch.rand(copies // 64, toks, d_model) * 2 - 1

    # Desired output is periodic in both token idx and d_model idx
    target_tok = torch.arange(d_model, dtype=torch.float) / d_model
    target = torch.zeros(toks, d_model)
    for i in range(toks):
        target[i] = target_tok * i % 1
    train_targets = target.repeat(copies, 1, 1)
    test_targets = target.repeat(copies // 64, 1, 1)

    model = NystromformerLM(
        max_tokens=toks,
        dim=d_model,
        depth=layers,
        dim_head=d_model,
        heads=4,
        num_landmarks=64,
        attn_dropout=0.1,
        ff_dropout=0.1,
    )

    dl_train = DataLoader(
        TensorDataset(train_data, train_targets), pin_memory=True, batch_size=128
    )
    dl_test = DataLoader(
        TensorDataset(test_data, test_targets), pin_memory=True, batch_size=128
    )

    trainer = pl.Trainer(gpus=1, max_epochs=10, precision=16)
    trainer.fit(model, dl_train)

    test_result = trainer.test(model, dl_test)
    assert test_result[0]["test/loss"] < 0.02
