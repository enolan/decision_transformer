import clip
import math
from omegaconf import OmegaConf
import pytorch_lightning as pl
from taming.models import vqgan
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class DecisionTransformer(pl.LightningModule):
    def __init__(
        self, d_model, n_head, n_layers, clip_model, vqgan_model, output_resolution
    ):
        super().__init__()

        self.vqgan_model = vqgan_model

        self.d_model = d_model
        # A linear layer embedding CLIP embeddings into our space.
        self.clip_embedding_linear = torch.nn.Linear(
            clip_model.visual.output_dim, d_model
        )
        # A linear layer embedding CLIP similarities
        self.clip_similarity_linear = torch.nn.Linear(1, d_model)
        # Embedding for VQGAN tokens
        self.vqgan_embedding = torch.nn.Embedding(vqgan_model.quantize.n_e, d_model)

        # How wide are the VQGAN patches?
        vqgan_token_size = 2 ** (vqgan_model.decoder.num_resolutions - 1)
        assert output_resolution % vqgan_token_size == 0
        self.vqgan_tokens = (output_resolution // vqgan_token_size) ** 2

        self.setup_positional_encoding()

        transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model, n_head)
        self.encoder = torch.nn.TransformerEncoder(transformer_encoder_layer, n_layers)

        self.decoder_vqgan_tokens = torch.nn.Linear(d_model, vqgan_model.quantize.n_e)
        self.decoder_cos_sim = torch.nn.Linear(d_model, 1)

    def training_step(self, batch, batch_idx):
        # print(list(self.named_buffers()))

        targets, cos_sims, tokenses = batch
        targets_e = self.clip_embedding_linear(targets)
        cos_sims_e = self.clip_similarity_linear(cos_sims)
        tokenses_e = self.vqgan_embedding(tokenses)
        # print(f"targets {targets_e.shape} cos_sims {cos_sims_e.shape} tokenses_e {tokenses_e.shape}")

        inputs = torch.cat(
            [targets_e.unsqueeze(1), cos_sims_e.unsqueeze(1), tokenses_e], axis=1
        )
        # print(f"inputs {inputs.shape}")

        # print(inputs.device)
        # print(self.positional.device)
        inputs = inputs + self.positional
        # print(inputs)

        encoded = self.encoder(inputs)
        # we have no loss for the target CLIP embedding since we never want to learn it
        vqgan_probs = self.decoder_vqgan_tokens(encoded)[
            :, 2:
        ]  # logged and unnormalized probabilities
        cos_sim_pred = self.decoder_cos_sim(encoded)[:, 1]
        # print(f"encoded {encoded.shape}, vqgan_probs {vqgan_probs.shape}, cos_sim_pred {cos_sim_pred.shape}")

        patches_loss = F.cross_entropy(
            vqgan_probs.reshape(-1, self.vqgan_model.quantize.n_e), tokenses.reshape(-1)
        )
        cos_sim_loss = F.mse_loss(cos_sim_pred, cos_sims)

        return 2 * cos_sim_loss + patches_loss  # mess with the scaling constant?

    def setup_positional_encoding(self):
        # Output tensor is (vqgan_tokens + 2, d_model)
        # Two extra tokens for the target CLIP embedding and the cosine
        # similarity, treated specially.
        # order might be wrong?

        self.register_buffer(
            "positional", torch.zeros(self.d_model).repeat(self.vqgan_tokens + 2, 1)
        )
        self.positional[0][0] = 1.0
        self.positional[1][1] = 1.0

        token_idxs = torch.arange(self.vqgan_tokens).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model - 2, 2)
            * (-math.log(10000.0) / (self.d_model - 2))
        )
        pos_e = torch.zeros(self.vqgan_tokens, self.d_model - 2)
        pos_e[:, 0::2] = torch.sin(token_idxs * div_term)
        pos_e[:, 1::2] = torch.cos(token_idxs * div_term)
        self.positional[2:, 2:] = pos_e

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


class TrainLossRecorder(pl.Callback):
    def __init__(self):
        self.train_loss = None

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.train_loss = outputs["loss"].item()


def setup_clip_and_vqgan(want_vqgan_weights=True):
    clip_model, _clip_preprocessor = clip.load("ViT-B/32")
    vqgan_config = OmegaConf.load("models/vqgan_imagenet_f16_16384.yaml")
    vqgan_model = vqgan.VQModel(**vqgan_config.model.params)
    if want_vqgan_weights:
        vqgan_model.init_from_ckpt("models/vqgan_imagenet_f16_16384.ckpt")
    return clip_model, vqgan_model


def test_can_init_dt():
    "Check setting up a DT module works"
    clip_model, vqgan_model = setup_clip_and_vqgan(want_vqgan_weights=False)
    DecisionTransformer(16, 4, 4, clip_model, vqgan_model, 64)


def test_dummy_train():
    clip_model, vqgan_model = setup_clip_and_vqgan(want_vqgan_weights=False)
    dt_model = DecisionTransformer(16, 4, 4, clip_model, vqgan_model, 64)
    dummy_clip_target = torch.zeros(clip_model.visual.output_dim)

    copies = 4096

    dummy_clip_targets = dummy_clip_target.unsqueeze(0).repeat(copies, 1)
    dummy_cos_sims = torch.ones(1).unsqueeze(0).repeat(copies, 1)
    dummy_vqgan_tokens = (
        torch.zeros(16, dtype=torch.long).unsqueeze(0).repeat(copies, 1)
    )
    dl = DataLoader(
        TensorDataset(dummy_clip_targets, dummy_cos_sims, dummy_vqgan_tokens),
        pin_memory=True,
        batch_size=512,
    )

    loss_recorder = TrainLossRecorder()
    trainer = pl.Trainer(gpus=1, max_epochs=100, callbacks=[loss_recorder])
    trainer.fit(dt_model, dl)

    assert loss_recorder.train_loss < 0.01
