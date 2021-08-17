import clip
import inspect
import math
from omegaconf import OmegaConf
import os
from pathlib import Path
import PIL
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
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
from tqdm import tqdm
import wandb

from clip_gen.positional_encoding import gen_positional_encoding
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
        config,
        clip_model,
        vqgan_model,
    ):
        super().__init__()
        self.save_hyperparameters(config)

        wandb.config.model_version = type(self).__name__
        wandb.config.update(config)

        self.vqgan_model = vqgan_model
        self.clip_model = clip_model
        self.d_model = config["d_model"]

        # A linear layer embedding CLIP embeddings into our space.
        self.clip_embedding_linear = torch.nn.Linear(
            clip_model.visual.output_dim, self.d_model, bias=False
        )
        # A linear layer embedding CLIP similarities
        self.clip_similarity_linear = torch.nn.Linear(1, self.d_model, bias=False)
        # Embedding for VQGAN tokens
        self.vqgan_embedding = torch.nn.Embedding(
            vqgan_model.quantize.n_embed, self.d_model
        )

        # How wide are the VQGAN patches?
        vqgan_token_size = 2 ** (vqgan_model.decoder.num_resolutions - 1)
        assert config["output_resolution"] % vqgan_token_size == 0
        # How many VQGAN tokens are there in an image?
        self.vqgan_tokens = (config["output_resolution"] // vqgan_token_size) ** 2

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
            config["d_model"],
            config["n_head"],
            batch_first=True,
            activation="gelu",
            dim_feedforward=config["dim_feedforward"],
        )
        self.encoder = torch.nn.TransformerEncoder(
            transformer_encoder_layer, config["n_layers"]
        )

        self.decoder_vqgan_tokens = torch.nn.Linear(
            self.d_model, vqgan_model.quantize.n_embed, bias=False
        )
        self.decoder_cos_sim = torch.nn.Linear(self.d_model, 1)

    def training_step(self, batch, batch_idx):
        imgs_tensor, _ = batch
        batch_size = imgs_tensor.shape[0]

        vqgan_tokenses, reconstructed_imgs = self._encode_with_vqgan(imgs_tensor)

        # Process with CLIP
        # TODO should I move it back to (0, 1) before the normalization step?
        # Probably.
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

        # TODO make the probability that a clip target is shuffled to the same
        # position it started in independent of batch size.
        clip_fuzzed_targets = clip_fuzzed_targets[torch.randperm(batch_size)]

        cos_sims = torch.nn.functional.cosine_similarity(
            clip_embeddings, clip_embeddings  # clip_fuzzed_targets
        ).unsqueeze(1)

        self.logger.experiment.log(
            {
                "train.cos_sim_in": wandb.Histogram(cos_sims[:, 0].detach().cpu()),
                "global_step": self.global_step,
            }
        )

        vqgan_tokenses = vqgan_tokenses.reshape(batch_size, -1)
        targets_e = self.clip_embedding_linear(clip_embeddings)
        cos_sims_e = self.clip_similarity_linear(cos_sims)
        tokenses_e = self.vqgan_embedding(vqgan_tokenses)

        inputs = torch.cat(
            [targets_e.unsqueeze(1), cos_sims_e.unsqueeze(1), tokenses_e], axis=1
        )

        inputs = inputs + self.positional

        encoded = self.encoder(inputs, mask=self.attn_mask)

        # all predictions are offset by 1. I.e. output n is a prediction for
        # token n + 1
        cos_sim_pred = self.decoder_cos_sim(encoded)[:, 0]
        self.logger.experiment.log(
            {
                "train.cos_sim_pred": wandb.Histogram(cos_sim_pred.detach().cpu()),
                "global_step": self.global_step,
            }
        )
        cos_sim_loss = F.mse_loss(cos_sim_pred, cos_sims)
        self.log("train.cos_sim_loss", cos_sim_loss)

        patches_loss = self._compute_vqgan_loss(encoded)

        # we have no loss for the target CLIP embedding since we never want
        # to learn it

        loss = patches_loss  # cos_sim_loss + patches_loss  # mess with the scaling constant?
        assert not loss.isnan().item()
        self.log("train.loss", loss)
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
            self.logger.experiment.log(
                {
                    "sampled_original_img": wandb.Image(imgs[idx]),
                    "sampled_reconstructed_img": wandb.Image(
                        ((reconstructed_imgs[idx] + 1) / 2).clamp(0, 1)
                    ),
                    "global_step": self.global_step,
                }
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

        self.logger.experiment.log(
            {
                "train.patch_losses": wandb.Histogram(patch_losses.detach().cpu()),
                "global_step": self.global_step,
            }
        )

        patches_loss = patch_losses.mean()
        self.log("train.patches_loss", patches_loss)

        return patches_loss

    def _enable_dropout(self):
        # Enable dropout for MC sampling
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def forward(self, target_clip_embedding, text=None):
        self._enable_dropout()

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

    def _sample_img(self, toks, top_p):
        """Sample an image given a vector of input tokens with everything
        before the VQGAN tokens filled in. Mutates toks.
        SHAPES:
        toks - (1, vqgan_tokens + 2, d_model)
        return - (vqgan_tokens)
        """

        vqgan_toks = []

        for i in range(self.vqgan_tokens):
            encoded = self.encoder(toks, mask=self.attn_mask)[0, i + 1]
            vqgan_token_probabilities = self.decoder_vqgan_tokens(encoded.unsqueeze(0))[
                0
            ]
            vqgan_token_probabilities = vqgan_token_probabilities.softmax(0)

            filtered_token_probabilities = self._filter_top_p(
                vqgan_token_probabilities, top_p
            )

            sampled_tok = filtered_token_probabilities.multinomial(1)[0]
            toks[0, i + 2] = toks[0, i + 2] + self.vqgan_embedding(sampled_tok)
            vqgan_toks.append(sampled_tok)

        return torch.stack(vqgan_toks)

    def _filter_top_p(self, probabilities, p):
        """Given a 1-d tensor of proabilities, return a tensor with the top p
        selected and rescaled and all others set to zero.
        """
        if p < 1.0:
            # Indexing into Torch tensors is hideously slow for some reason:
            # https://github.com/pytorch/pytorch/issues/29973. So we copy them
            # to CPU and use numpy.
            probabilities = np.asarray(probabilities.cpu())
            sort_idxs = np.argsort(probabilities)[::-1]
            probability_so_far = 0.0
            filtered_token_probabilities = np.zeros(probabilities.shape[0])
            for i in sort_idxs:
                probability_so_far = probability_so_far + probabilities[i]
                filtered_token_probabilities[i] = probabilities[i]
                if probability_so_far >= p:
                    break
            filtered_token_probabilities = torch.tensor(
                filtered_token_probabilities / probability_so_far, device=self.device
            )
        else:
            filtered_token_probabilities = vqgan_token_probabilities
        return filtered_token_probabilities

    def _sample_img_without_autoregression(self, toks):
        """Same as _sample_img, but does not generate the image progressively
        adding each output patch into the input as it works. Does not mutate
        toks."""

        encoded = self.encoder(toks, mask=self.attn_mask)

        vqgan_token_probabilities = self.decoder_vqgan_tokens(encoded)[0, 1:-1]
        vqgan_token_probabilities = vqgan_token_probabilities.softmax(1)
        vqgan_tokens = torch.distributions.Categorical(
            probs=vqgan_token_probabilities
        ).sample()

        return vqgan_tokens

    def setup_positional_encoding(self):
        # Output tensor is (vqgan_tokens + 2, d_model)
        # Originally I had special encodings for the CLIP target and similarity
        # but I nixed them for simplicity. Probably doesn't matter.
        self.register_buffer(
            "positional",
            gen_positional_encoding(self.d_model, self.vqgan_tokens + 2),
            persistent=False,
        )

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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

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

    def on_save_checkpoint(self, checkpoint):
        out_state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            if not (k.startswith("clip_model") or k.startswith("vqgan_model")):
                out_state_dict[k] = v
        checkpoint["state_dict"] = out_state_dict


def init_wandb_for_test(record=True):
    if record:
        mode = "online"
    else:
        mode = "disabled"
    wandb.init(
        reinit=True,
        mode=mode,
        job_type="unit_test",
        config={"test_name": inspect.stack()[1][3]},
    )
    return WandbLogger()


def test_can_init_dt():
    "Check setting up a DT module works"
    clip_model, _clip_preprocessor, vqgan_model = setup_clip_and_vqgan(
        want_vqgan_weights=False
    )
    init_wandb_for_test(record=False)
    DecisionTransformer(config_tiny, clip_model, vqgan_model)


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

    def forward(self, _target_clip_embedding, text=None, top_p=1.0):
        inputs = torch.zeros(1, self.vqgan_tokens + 2, self.d_model, device=self.device)

        vqgan_tokens = self._sample_img_without_autoregression(inputs)

        return self._vqgan_toks_to_image(vqgan_tokens)


def test_no_input():
    clip_model, _clip_preprocessor, vqgan_model = setup_clip_and_vqgan(
        want_vqgan_weights=True
    )

    logger = init_wandb_for_test()
    cfg = config_tiny | {"lr": 5e-3}
    model = NoInputModel(cfg, clip_model, vqgan_model)

    dl = setup_dataloader("debug_test_data/all-white", 64, cfg["output_resolution"])

    eval_callback = EvalEveryNIts(["a white square"], model, 100)
    loss_recorder = TrainLossRecorder()
    trainer = pl.Trainer(
        gpus=1,
        log_every_n_steps=1,
        max_steps=100,
        precision=16,
        callbacks=[eval_callback, loss_recorder, CheckGradients(generate_charts=False)],
        logger=logger,
    )
    trainer.fit(model, dl)
    assert loss_recorder.train_loss < 3.8


class PositionalOnlyModel(DecisionTransformer):
    """Class to test model with only the positional encoding. Should be able to
    memorize a single image with arbitrary structure, but not multiple since it
    doesn't know what else has been drawn. Interestingly it sort of appears to
    if you give it a dataset with more than 1 item since the VQGAN encoding
    involves local context.
    """

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        batch_size = imgs.shape[0]

        vqgan_tokens, _ = self._encode_with_vqgan(imgs)
        vqgan_tokens = vqgan_tokens.reshape(batch_size, -1)

        inputs = self.positional.expand(batch_size, self.vqgan_tokens + 2, self.d_model)

        encoded = self.encoder(inputs, mask=self.attn_mask)

        patches_loss = self._compute_vqgan_loss(encoded, vqgan_tokens)

        return patches_loss

    def forward(self, _target_clip_embedding, text=None, top_p=1.0):
        inputs = self.positional.unsqueeze(0)
        vqgan_tokens = self._sample_img_without_autoregression(inputs)
        return self._vqgan_toks_to_image(vqgan_tokens)


def test_positional_only():
    clip_model, _clip_preprocessor, vqgan_model = setup_clip_and_vqgan(
        want_vqgan_weights=True
    )

    logger = init_wandb_for_test()
    cfg = config_small
    model = PositionalOnlyModel(cfg, clip_model, vqgan_model)

    dl = setup_dataloader("debug_test_data/4 colors", 64, cfg["output_resolution"])
    eval_callback = EvalEveryNIts(["a four color square"], model, 100)
    loss_recorder = TrainLossRecorder()
    trainer = pl.Trainer(
        gpus=1,
        log_every_n_steps=1,
        max_steps=250,
        precision=16,
        callbacks=[eval_callback, loss_recorder, CheckGradients(clip_percentile=0.95)],
        logger=logger,
    )
    trainer.fit(model, dl)
    assert loss_recorder.train_loss < 1.2


class PositionalAndAutoregressiveModel(DecisionTransformer):
    """Class to test model with both the positional encoding and access to
    previously generated tokens. Should be able to model an arbitrary dataset
    but not do any of the CLIP conditioning stuff."""

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        batch_size = imgs.shape[0]

        vqgan_tokens, _ = self._encode_with_vqgan(imgs)
        vqgan_tokens = vqgan_tokens.reshape(batch_size, -1)

        inputs = self.positional.repeat(batch_size, 1, 1)

        vqgan_tokens_encoded = self.vqgan_embedding(vqgan_tokens)
        inputs[:, 2:] = vqgan_tokens_encoded + inputs[:, 2:]

        encoded = self.encoder(inputs, mask=self.attn_mask)

        patches_loss = self._compute_vqgan_loss(encoded, vqgan_tokens)

        return patches_loss

    def forward(self, _target_clip_embedding, text=None, top_p=1.0):
        self._enable_dropout()

        toks = torch.clone(self.positional).unsqueeze(0)
        vqgan_toks = self._sample_img(toks, top_p)

        return self._vqgan_toks_to_image(vqgan_toks)


def test_positional_and_autoregressive():
    clip_model, _clip_preprocessor, vqgan_model = setup_clip_and_vqgan(
        want_vqgan_weights=True
    )

    logger = init_wandb_for_test()
    cfg = config_medium_small
    model = PositionalAndAutoregressiveModel(cfg, clip_model, vqgan_model)

    dl = setup_dataloader(
        "debug_test_data/diverse",  # "diverse" means 32 copies of 32 images
        64,
        cfg["output_resolution"],
    )
    eval_callback = EvalEveryNIts(["a four color square"], model, 500)
    loss_recorder = TrainLossRecorder()
    trainer = pl.Trainer(
        gpus=1,
        log_every_n_steps=1,
        max_steps=9_000,
        precision=16,
        callbacks=[
            eval_callback,
            loss_recorder,
            CheckGradients(clip_percentile=0.95),
        ],
        logger=logger,
    )
    trainer.fit(model, dl)
    assert loss_recorder.train_loss < 1.85


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
            print(f"generating eval images at {pl_module.global_step}")
            with torch.no_grad():
                pl_module.eval()
                for prompt in tqdm(self.prompts, leave=False):
                    log_dict = {"global_step": pl_module.global_step}
                    for p in tqdm([0.5, 0.75, 0.85, 0.95], leave=False):
                        imgs = []
                        for _ in range(2):
                            imgs.append(
                                wandb.Image(
                                    pl_module(
                                        prompt["embedding"],
                                        text=prompt["prompt"],
                                        top_p=p,
                                    )
                                )
                            )
                        log_dict[f"eval-images.{prompt['prompt']}.top-{p:.2f}"] = imgs
                    pl_module.logger.experiment.log(log_dict)
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
            torchvision.transforms.RandomCrop(smaller_dim),
            torchvision.transforms.Resize((target_res, target_res)),
        ]
    )
    return torchvision.transforms.functional.to_tensor(transform(img))


config_tiny = {
    "d_model": 16,
    "n_head": 4,
    "dim_feedforward": 16,
    "n_layers": 4,
    "output_resolution": 64,
    "lr": 1e-3,
}

config_small = {
    "d_model": 64,
    "n_head": 4,
    "dim_feedforward": 2048,
    "n_layers": 4,
    "output_resolution": 64,
    "lr": 1e-3,
}

config_medium_small = {
    "d_model": 256,
    "n_head": 8,
    "dim_feedforward": 2048,
    "n_layers": 8,
    "output_resolution": 64,
    "lr": 3e-4,
}

# stole these hyperparameters from GPT-1, modulo the layer count which is 12 in
# their implementation
config_medium = {
    "d_model": 768,
    "n_head": 12,
    "dim_feedforward": 3072,
    "n_layers": 8,
    "output_resolution": 128,
    "lr": 1e-3,  # lr *not* from GPT
}


def setup_dataloader(path, batch_size, output_res):
    wandb.config.update({"data_path": path, "batch_size": batch_size})
    return DataLoader(
        FilteredImageFolder(
            path,
            transform=lambda img: transform_image(img, output_res),
        ),
        pin_memory=True,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )


if __name__ == "__main__":

    matplotlib.use("svg")

    clip_model, clip_preprocessor, vqgan_model = setup_clip_and_vqgan(
        want_vqgan_weights=True
    )

    cfg = config_medium

    wandb.init()
    dt_model = PositionalAndAutoregressiveModel(
        cfg,
        clip_model,
        vqgan_model,
    )

    wandb.watch(dt_model, log_freq=100)
    wandb_logger = WandbLogger()

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
        1000,
    )

    dl = setup_dataloader(
        "/home/enolan/mystuff/code/clip-gen/debug_test_data/cats",
        16,
        cfg["output_resolution"],
    )

    trainer = pl.Trainer(
        gpus=1,
        log_every_n_steps=1,
        callbacks=[
            eval_callback,
            CheckGradients(clip_percentile=0.95),
        ],
        precision=16,
        max_steps=100_000,
        logger=wandb_logger,
    )
    trainer.fit(dt_model, dl)
