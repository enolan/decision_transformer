import clip
import math
from omegaconf import OmegaConf
import os
from pathlib import Path
import PIL
import pytorch_lightning as pl
import matplotlib
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
import numpy as np
import random
from taming.models import vqgan
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision

from clip_gen.utils import CheckGradients, FilteredImageFolder, TrainLossRecorder


class DecisionTransformer(pl.LightningModule):
    """Main modified decision transformer model.
    Basic idea is we use an autoregressive transformer to model sequences of:
    1. Target CLIP embedding
    2. Cosine similarity of image embedding with target
    3. Sequence of VQGAN patches

    Generating the images is then a matter of picking a target CLIP embedding,
    sampling to find a plausible cosine similarity, then sampling the series of
    tokens that encode the image.

    This is basically the same as Katherine Crowson's decision transformer
    model, the difference is that this is trained without the use of image
    labels - we do some "data augmentation" on unlabeled images to generate our
    training data. Hopefully this lets us get more diverse outputs without a
    need for manually labeled data.
    """

    def __init__(
        self,
        d_model,
        n_head,
        n_layers,
        clip_model,
        vqgan_model,
        output_resolution,
    ):
        super().__init__()

        self.vqgan_model = vqgan_model
        self.clip_model = clip_model
        self.d_model = d_model

        # A linear layer embedding CLIP embeddings into our space.
        self.clip_embedding_linear = torch.nn.Linear(
            clip_model.visual.output_dim, d_model, bias=False
        )
        # A linear layer embedding CLIP similarities
        self.clip_similarity_linear = torch.nn.Linear(1, d_model, bias=False)
        # Embedding for VQGAN tokens
        self.vqgan_embedding = torch.nn.Embedding(vqgan_model.quantize.n_embed, d_model)

        # How wide are the VQGAN patches?
        vqgan_token_size = 2 ** (vqgan_model.decoder.num_resolutions - 1)
        assert output_resolution % vqgan_token_size == 0
        # How many VQGAN tokens are there in an image?
        self.vqgan_tokens = (output_resolution // vqgan_token_size) ** 2

        self.setup_positional_encoding()
        self.setup_attention_mask()
        self.normalize_for_clip = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    [
                        clip_model.visual.input_resolution,
                        clip_model.visual.input_resolution,
                    ]
                ),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        transformer_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model, n_head, batch_first=True, activation="gelu"
        )
        self.encoder = torch.nn.TransformerEncoder(transformer_encoder_layer, n_layers)

        self.decoder_vqgan_tokens = torch.nn.Linear(
            d_model, vqgan_model.quantize.n_embed, bias=False
        )
        self.decoder_cos_sim = torch.nn.Linear(d_model, 1)

    def training_step(self, batch, batch_idx):
        imgs_tensor, _ = batch
        batch_size = imgs_tensor.shape[0]

        vqgan_tokenses, reconstructed_imgs = self._encode_with_vqgan(imgs_tensor)

        # Process with CLIP
        reconstructed_for_clip = self.normalize_for_clip(reconstructed_imgs)

        clip_embeddings = self.clip_model.encode_image(reconstructed_for_clip).type_as(
            reconstructed_for_clip
        )

        # We add gaussian noise to the CLIP target vectors and shuffle them in
        # order to force the model to learn the CLIP embedding ->
        # (image, cos similarity) function.
        clip_fuzzed_targets = (
            clip_embeddings
            + torch.distributions.normal.Normal(
                torch.zeros(
                    [batch_size, self.clip_model.visual.output_dim], device=self.device
                ),
                torch.tensor([0.2], device=self.device),
            ).sample()
        )
        clip_fuzzed_targets = clip_fuzzed_targets / torch.linalg.vector_norm(
            clip_fuzzed_targets, dim=1, keepdim=True
        )

        clip_fuzzed_targets = clip_fuzzed_targets[torch.randperm(batch_size)]

        cos_sims = torch.nn.functional.cosine_similarity(
            clip_embeddings, clip_embeddings  # clip_fuzzed_targets
        ).unsqueeze(1)

        self.logger.experiment.add_histogram(
            "train/cos_sim_in", cos_sims[:, 0], global_step=self.global_step
        )

        vqgan_tokenses = vqgan_tokenses.reshape(batch_size, -1)
        targets_e = self.clip_embedding_linear(clip_embeddings)
        cos_sims_e = self.clip_similarity_linear(cos_sims)
        tokenses_e = self.vqgan_embedding(vqgan_tokenses)

        inputs = torch.cat(
            [targets_e.unsqueeze(1), cos_sims_e.unsqueeze(1), tokenses_e], axis=1
        )

        # for tok_idx in range(0, self.vqgan_tokens + 2, (self.vqgan_tokens + 2) // 8):
        for tok_idx in range(0, self.vqgan_tokens + 2):
            self.logger.experiment.add_histogram(
                f"position {tok_idx} positional channels",
                self.positional[tok_idx, :],
                global_step=self.global_step,
            )

        inputs = inputs + self.positional

        encoded = self.encoder(inputs, mask=self.attn_mask)

        # all predictions are offset by 1. I.e. output n is a prediction for
        # token n + 1
        cos_sim_pred = self.decoder_cos_sim(encoded)[:, 0]
        self.logger.experiment.add_histogram(
            "train/cos_sim_pred", cos_sim_pred, global_step=self.global_step
        )
        cos_sim_loss = F.mse_loss(cos_sim_pred, cos_sims)
        self.log("train/cos_sim_loss", cos_sim_loss)

        patches_loss = self._compute_vqgan_loss(encoded)

        # we have no loss for the target CLIP embedding since we never want
        # to learn it

        loss = patches_loss  # cos_sim_loss + patches_loss  # mess with the scaling constant?
        assert not loss.isnan().item()
        self.log("train/loss", loss)
        return loss

    def _encode_with_vqgan(self, imgs):
        """Encode a tensor of images (in range (0, 1)) with the VQGAN,
        returning both the VQGAN tokens and the reconstructed image (this time
        in range (-1, 1)).
        SHAPES:
        - imgs (batch, channels, height, width)
        - returned vqgan tokens (batch, height, width)
        - returned imgs (batch, channels, height, width)
        """
        # VQGAN does not play well with mixed precision :(
        with torch.cuda.amp.autocast(False):
            self.vqgan_model.eval()
            vqgan_zs, _embedding_loss, [_, _, vqgan_tokens] = self.vqgan_model.encode(
                imgs * 2 - 1
            )
            assert not vqgan_zs.isnan().any().item()

            reconstructed_imgs = self.vqgan_model.decode(vqgan_zs)

        # Stick a reconstructed example in TensorBoard for debug purposes
        if random.random() <= 0.01:
            idx = random.randint(0, imgs.shape[0] - 1)
            self.logger.experiment.add_image(
                f"sampled_original_img", imgs[idx], global_step=self.global_step
            )
            self.logger.experiment.add_image(
                f"sampled_reconstructed_img",
                ((reconstructed_imgs[idx] + 1) / 2).clamp(0, 1),
                global_step=self.global_step,
            )
        return vqgan_tokens, reconstructed_imgs

    def _compute_vqgan_loss(self, encoder_output, target_vqgan_tokens):
        """Compute the loss for VQGAN tokens. encoder_output should come from
        the transformer encoder, target_vqgan_tokens from the examples we're
        training on.
        SHAPES:
        - encoder_output (batch, token, d_model)
        - target_vqgan_tokens (batch, vqgan token) - tokens should be ints
        """

        vqgan_probs = self.decoder_vqgan_tokens(encoder_output)[:, 1:-1]
        patch_losses = []

        for i in range(vqgan_probs.shape[1]):
            patch_losses.append(
                F.cross_entropy(
                    vqgan_probs[:, i, :], target_vqgan_tokens[:, i]
                ).unsqueeze(0)
            )
        patch_losses = torch.cat(patch_losses)

        self.logger.experiment.add_histogram(
            "train/patch_losses", patch_losses, global_step=self.global_step
        )

        patches_loss = patch_losses.mean()
        self.log("train/patches_loss", patches_loss)

        return patches_loss

    def forward(self, target_clip_embedding, text=None):
        # Enable dropout for MC sampling
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

        target_e = self.clip_embedding_linear(target_clip_embedding)

        toks = torch.clone(self.positional).unsqueeze(0)
        toks[0][0] = target_e + toks[0][0]

        cos_sim_candidates = []
        for _ in range(100):  # How many samples do we need? 100 is asspulled
            encoded = self.encoder(toks, mask=self.attn_mask)[0, 0]
            pred_cos_sim = self.decoder_cos_sim(encoded).clamp(0, 1)
            cos_sim_candidates.append(pred_cos_sim)
        cos_sim_candidates, _idxs = torch.cat(cos_sim_candidates).sort()

        # 90th percentile is asspulled too
        cos_sim_target = cos_sim_candidates[int(len(cos_sim_candidates) * 0.90)]

        encoded_cos_sim = self.clip_similarity_linear(cos_sim_target.unsqueeze(0))
        toks[0, 1] = toks[0, 1] + encoded_cos_sim

        vqgan_toks = self._sample_img(toks)

        return self._vqgan_toks_to_image(vqgan_toks)

    def _sample_img(self, toks):
        """Sample an image given a vector of input tokens with everything
        before the VQGAN tokens filled in. Mutates toks.
        SHAPES:
        toks - (1, vqgan_tokens + 2, d_model)
        return - (vqgan_tokens)
        """

        vqgan_toks = []

        for i in range(self.vqgan_tokens):
            # Could do top-p sampling here, or one of the other ones. Doing the
            # simplest approach for now.
            encoded = self.encoder(toks, mask=self.attn_mask)[0, i + 1]
            vqgan_token_probabilities = self.decoder_vqgan_tokens(encoded.unsqueeze(0))[
                0
            ]
            vqgan_token_probabilities = vqgan_token_probabilities.softmax(0)
            sampled_tok = vqgan_token_probabilities.multinomial(1)[0]
            toks[0, i + 2] = self.vqgan_embedding(sampled_tok)
            vqgan_toks.append(sampled_tok)

        return torch.stack(vqgan_toks)

    def setup_positional_encoding(self):
        # Output tensor is (vqgan_tokens + 2, d_model)
        # Two extra tokens for the target CLIP embedding and the cosine
        # similarity, treated specially.

        positional = torch.nn.parameter.Parameter(
            torch.distributions.uniform.Uniform(-1.0, 1.0).rsample(
                (self.vqgan_tokens + 2, self.d_model)
            )
        )
        self.register_parameter("positional", positional)

    def setup_attention_mask(self):
        self.register_buffer(
            "attn_mask",
            torch.triu(
                torch.ones(self.vqgan_tokens + 2, self.vqgan_tokens + 2)
                * float("-inf"),
                diagonal=1,
            ),
            persistent=False,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def _vqgan_toks_to_image(self, vqgan_toks):
        """Given a tensor of VQGAN tokens, return an image with shape
        (channels, height, width) suitable for sending to Tensorboard.
        """
        img_res_in_tokens = int(math.sqrt(self.vqgan_tokens))

        z = self.vqgan_model.quantize.embed(vqgan_toks)
        z = z.reshape((1, img_res_in_tokens, img_res_in_tokens, -1))
        z = z.transpose(1, 3)
        z = z.transpose(2, 3)
        img = ((self.vqgan_model.decode(z)[0] + 1) / 2).clamp(0, 1)
        return img


def test_can_init_dt():
    "Check setting up a DT module works"
    clip_model, _clip_preprocessor, vqgan_model = setup_clip_and_vqgan(
        want_vqgan_weights=False
    )
    DecisionTransformer(16, 4, 4, clip_model, vqgan_model, 64)


class NoInputModel(DecisionTransformer):
    """Class to test model with 0 inputs. It should be able to memorize an image
    that is all one color, but not track different positions differently or
    memorize multiple images."""

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        batch_size = imgs.shape[0]

        vqgan_tokens, _ = self._encode_with_vqgan(imgs)
        vqgan_tokens = vqgan_tokens.reshape(batch_size, -1)

        inputs = torch.zeros(
            batch_size, self.vqgan_tokens + 2, self.d_model, device=self.device
        )

        encoded = self.encoder(inputs, mask=self.attn_mask)

        patches_loss = self._compute_vqgan_loss(encoded, vqgan_tokens)

        return patches_loss

    def forward(self, _target_clip_embedding, text=None):
        inputs = torch.zeros(1, self.vqgan_tokens + 2, self.d_model, device=self.device)

        encoded = self.encoder(inputs, mask=self.attn_mask)

        vqgan_token_probabilities = self.decoder_vqgan_tokens(encoded)[0, 1:-1]
        vqgan_token_probabilities = vqgan_token_probabilities.softmax(1)
        vqgan_tokens = torch.distributions.Categorical(
            probs=vqgan_token_probabilities
        ).sample()

        return self._vqgan_toks_to_image(vqgan_tokens)


def test_no_input():
    clip_model, _clip_preprocessor, vqgan_model = setup_clip_and_vqgan(
        want_vqgan_weights=True
    )

    model_res = 64
    model = NoInputModel(32, 4, 4, clip_model, vqgan_model, model_res)

    dl = DataLoader(
        torchvision.datasets.ImageFolder(
            "debug_test_data/all-white",
            transform=lambda img: transform_image(img, model_res),
        ),
        pin_memory=True,
        batch_size=64,
        num_workers=8,
    )

    eval_callback = EvalEveryNIts(["a white square"], model, 100)
    loss_recorder = TrainLossRecorder()
    trainer = pl.Trainer(
        gpus=1,
        log_every_n_steps=1,
        max_steps=150,
        precision=16,
        callbacks=[eval_callback, loss_recorder, CheckGradients(generate_charts=False)],
    )
    trainer.fit(model, dl)
    assert loss_recorder.train_loss < 3.8


class EvalEveryNIts(pl.Callback):
    def __init__(self, prompts, model, n):
        self.n = n
        self.prompts = []
        with torch.no_grad():
            for prompt in prompts:
                tokens = clip.tokenize([prompt]).cuda()
                embedding = model.clip_model.encode_text(tokens).float()
                self.prompts.append({"prompt": prompt, "embedding": embedding})

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if pl_module.global_step % self.n == 0:
            with torch.no_grad():
                pl_module.eval()
                for prompt in self.prompts:
                    img = pl_module(prompt["embedding"], text=prompt["prompt"])
                    pl_module.logger.experiment.add_image(
                        f"{prompt['prompt']}", img, global_step=pl_module.global_step
                    )
                pl_module.train()


def setup_clip_and_vqgan(want_vqgan_weights=True):
    "Load the models we depend on."
    clip_model, clip_preprocessor = clip.load("ViT-B/32")
    clip_model.eval().requires_grad_(False)

    vqgan_config = OmegaConf.load("models/vqgan_gumbel_openimages_f8_8192.yaml")
    vqgan_model = vqgan.GumbelVQ(**vqgan_config.model.params)
    if want_vqgan_weights:
        vqgan_model.init_from_ckpt("models/vqgan_gumbel_openimages_f8_8192.ckpt")
    del vqgan_model.loss
    vqgan_model.eval().requires_grad_(False)
    vqgan_model.quantize.temperature = 1e-10

    return clip_model, clip_preprocessor, vqgan_model


def transform_image(img, target_res):
    "Transform a PIL image into the appropriate tensors"

    # Fit to size
    smaller_dim = min(img.width, img.height)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop(smaller_dim),
            #            torchvision.transforms.RandomCrop(smaller_dim),
            torchvision.transforms.Resize((target_res, target_res)),
        ]
    )
    return torchvision.transforms.functional.to_tensor(transform(img))


if __name__ == "__main__":

    matplotlib.use("svg")

    output_res = 16
    clip_model, clip_preprocessor, vqgan_model = setup_clip_and_vqgan(
        want_vqgan_weights=True
    )
    dt_model = DecisionTransformer(64, 4, 8, clip_model, vqgan_model, output_res)

    eval_callback = EvalEveryNIts(
        [
            "a smiling woman"
            # "a sad man's face",
            # "a group of women",
            # "a man giving a speech",
            # "a bouquet of roses",
            # "Manhattan at sunset #pentax67",
            # "a painting inspired by a 5-MeO-DMT trip",
            # "Burning Man 2018 #artcar #pentax67",
        ],
        dt_model,
        500,
    )

    dl = DataLoader(
        FilteredImageFolder(
            "/home/enolan/mystuff/code/clip-gen/debug_test_data/4 colors",
            transform=lambda img: transform_image(
                img, output_res, clip_model, clip_preprocessor, vqgan_model
            ),
        ),
        pin_memory=True,
        batch_size=1,
        num_workers=8,
        # shuffle=True,
    )

    trainer = pl.Trainer(
        gpus=1,
        log_every_n_steps=1,
        callbacks=[eval_callback, CheckGradients()],
        precision=16,
    )
    trainer.fit(dt_model, dl)
