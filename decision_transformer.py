import clip
import math
from omegaconf import OmegaConf
import os
import PIL
import pytorch_lightning as pl
import random
from taming.models import vqgan
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision


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
        self, d_model, n_head, n_layers, clip_model, vqgan_model, output_resolution
    ):
        super().__init__()

        self.vqgan_model = vqgan_model
        self.clip_model = clip_model
        self.d_model = d_model
        self.logged_reconstructions = 0

        # A linear layer embedding CLIP embeddings into our space.
        self.clip_embedding_linear = torch.nn.Linear(
            clip_model.visual.output_dim, d_model
        )
        # A linear layer embedding CLIP similarities
        self.clip_similarity_linear = torch.nn.Linear(1, d_model)
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
            d_model, n_head, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(transformer_encoder_layer, n_layers)

        self.decoder_vqgan_tokens = torch.nn.Linear(
            d_model, vqgan_model.quantize.n_embed
        )
        self.decoder_cos_sim = torch.nn.Linear(d_model, 1)

    def training_step(self, batch, batch_idx):
        imgs_tensor, _ = batch
        batch_size = imgs_tensor.shape[0]

        # Encode with VQGAN
        vqgan_zs, _embedding_loss, [_, _, vqgan_tokenses] = self.vqgan_model.encode(
            imgs_tensor * 2 - 1
        )

        reconstructed_imgs = self.vqgan_model.decode(vqgan_zs)

        # Stick a reconstructed example in TensorBoard for debug purposes
        if random.random() <= 0.05:
            idx = random.randint(0, batch_size - 1)
            self.logger.experiment.add_image(
                f"original/{self.logged_reconstructions}", imgs_tensor[idx]
            )
            self.logger.experiment.add_image(
                f"reconstructed/{self.logged_reconstructions}",
                ((reconstructed_imgs[idx] + 1) / 2).clamp(0, 1),
            )
            self.logged_reconstructions = self.logged_reconstructions + 1

        # Process with CLIP
        reconstructed_for_clip = self.normalize_for_clip(reconstructed_imgs)
        clip_embeddings = self.clip_model.encode_image(reconstructed_for_clip).type_as(
            reconstructed_for_clip
        )
        cos_sims = torch.ones(batch_size).type_as(clip_embeddings).unsqueeze(1)

        vqgan_tokenses = vqgan_tokenses.reshape(batch_size, -1)
        # TODO data augmentation, fuzz the target and compute cosine similarity
        # to fuzzed target
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
        cos_sim_loss = F.mse_loss(cos_sim_pred, cos_sims)
        self.log("train/cos_sim_loss", cos_sim_loss)

        vqgan_probs = self.decoder_vqgan_tokens(encoded)[
            :, 1:-1
        ]  # logged and unnormalized probabilities
        patches_loss = F.cross_entropy(
            vqgan_probs.reshape(-1, self.vqgan_model.quantize.n_embed),
            vqgan_tokenses.reshape(-1),
        )
        self.log("train/patches_loss", patches_loss)

        # we have no loss for the target CLIP embedding since we never want
        # to learn it

        assert not cos_sim_loss.isnan().item()
        assert not patches_loss.isnan().item()
        loss = 2 * cos_sim_loss + patches_loss  # mess with the scaling constant?
        assert not loss.isnan().item()
        self.log("train/loss", loss)
        return loss

    def forward(self, target_clip_embedding):
        with torch.no_grad():
            # Enable dropout for MC sampling
            for m in self.modules():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()

            target_e = self.clip_embedding_linear(target_clip_embedding)

            toks = self.positional.unsqueeze(0)
            toks[0][0] = target_e + toks[0][0]

            #print(f"toks {toks}")

            cos_sim_candidates = []
            for _ in range(100):  # How many samples do we need? 100 is asspulled
                encoded = self.encoder(toks, mask=self.attn_mask)[0, 0]
                pred_cos_sim = self.decoder_cos_sim(encoded)
                cos_sim_candidates.append(pred_cos_sim)
            cos_sim_candidates, _idxs = torch.cat(cos_sim_candidates).sort()
            #print(f"cos_sim_candidates {cos_sim_candidates}")

            # 90th percentile is asspulled too
            cos_sim_target = cos_sim_candidates[int(len(cos_sim_candidates) * 0.90)]
            #print(f"cos_sim_target {cos_sim_target}")

            toks[0, 1] = toks[0, 1] + self.decoder_cos_sim(cos_sim_target.unsqueeze(0))

            vqgan_toks = []

            for i in range(self.vqgan_tokens):
                #print(f"toks {toks}")

                # Could do top-p sampling here, or one of the other ones. Doing
                # the simplest approach for now.
                encoded = self.encoder(toks, mask=self.attn_mask)[0, i + 1]
                vqgan_token_probabilities = self.decoder_vqgan_tokens(encoded.unsqueeze(0))
                vqgan_token_probabilities = vqgan_token_probabilities.softmax(1)
                sampled_toks = vqgan_token_probabilities.multinomial(1)
                #print(f"sampled_toks {sampled_toks}")
                toks[0, i + 2] = self.vqgan_embedding(sampled_toks[0, 0])
                vqgan_toks.append(sampled_toks[0, 0])

            img_res_in_tokens = int(math.sqrt(self.vqgan_tokens))

            vqgan_toks = torch.stack(vqgan_toks)
            print(f"vqgan_toks {vqgan_toks}")
            z = self.vqgan_model.quantize.embed(vqgan_toks)
            print(f"z {z} {z.shape}")
            z = z.reshape((1, img_res_in_tokens, img_res_in_tokens, -1))
            z = z.transpose(1, 3)
            z = z.transpose(2, 3)
            print(f"z {z.shape}")
            return ((self.vqgan_model.decode(z)[0] + 1) / 2).clamp(0, 1)


    def setup_positional_encoding(self):
        # Output tensor is (vqgan_tokens + 2, d_model)
        # Two extra tokens for the target CLIP embedding and the cosine
        # similarity, treated specially.

        self.register_buffer(
            "positional",
            torch.zeros(self.d_model).repeat(self.vqgan_tokens + 2, 1),
            persistent=False,
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

    def setup_attention_mask(self):
        self.register_buffer(
            "attn_mask",
            torch.triu(
                torch.ones(self.vqgan_tokens + 2, self.vqgan_tokens + 2)
                * float("-inf"),
                diagonal=1
                # this seems wrong? but it makes everything NaN otherwise. It
                # means the first token output is allowed to attend to itself.
                # Which shouldn't matter in this application but is still
                # wrong.
            ),
            persistent=False,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


class TrainLossRecorder(pl.Callback):
    "Tool for recording the training loss."

    def __init__(self):
        self.train_loss = None

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.train_loss = outputs["loss"].item()


class EvalEveryEpoch(pl.Callback):
    def __init__(self, prompts, model):
        self.prompts = []

        with torch.no_grad():
            for prompt in prompts:
                tokens = clip.tokenize([prompt]).cuda()
                embedding = model.clip_model.encode_text(tokens).float()
                self.prompts.append({"prompt": prompt, "embedding": embedding})

        self.ctr = 0

    def on_train_epoch_start(self, trainer, pl):
        for prompt in self.prompts:
            img = pl.forward(prompt["embedding"])
            pl.logger.experiment.add_image(f"{prompt['prompt']}/{self.ctr}", img)
        self.ctr = self.ctr + 1


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

    return clip_model, clip_preprocessor, vqgan_model


def test_can_init_dt():
    "Check setting up a DT module works"
    clip_model, _clip_preprocessor, vqgan_model = setup_clip_and_vqgan(
        want_vqgan_weights=False
    )
    DecisionTransformer(16, 4, 4, clip_model, vqgan_model, 64)


def test_dummy_train():
    "Check we can memorize a trivial dataset."
    clip_model, _clip_preprocessor, vqgan_model = setup_clip_and_vqgan(
        want_vqgan_weights=False
    )
    dt_model = DecisionTransformer(16, 4, 4, clip_model, vqgan_model, 64)
    dummy_clip_target = torch.zeros(clip_model.visual.output_dim)

    copies = 4096

    dummy_imgs = torch.zeros(copies, 3, 64, 64)
    dummy_targets = torch.zeros(copies, 1)

    dl = DataLoader(
        TensorDataset(dummy_imgs, dummy_targets),
        pin_memory=True,
        batch_size=16,
    )

    loss_recorder = TrainLossRecorder()
    trainer = pl.Trainer(gpus=1, max_epochs=35, callbacks=[loss_recorder])
    trainer.fit(dt_model, dl)

    assert loss_recorder.train_loss < 0.01


def transform_image(img, target_res, clip_model, clip_preprocessor, vqgan_model):
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


def clean_img_dir(dir):
    for path in os.listdir(dir):
        try:
            PIL.Image.open(f"{dir}/{path}").copy()
        except (PIL.UnidentifiedImageError, OSError):
            print(f"Deleting {path}")
            os.remove(f"{dir}/{path}")


if __name__ == "__main__":

    output_res = 256
    clip_model, clip_preprocessor, vqgan_model = setup_clip_and_vqgan(
        want_vqgan_weights=True
    )
    dt_model = DecisionTransformer(64, 4, 4, clip_model, vqgan_model, output_res)

    eval_callback = EvalEveryEpoch(
        ["a sad man's face", "a group of women", "a man giving a speech"], dt_model
    )

    dl = DataLoader(
        torchvision.datasets.ImageFolder(
            "/run/user/1000/learndata",
            transform=lambda img: transform_image(
                img, output_res, clip_model, clip_preprocessor, vqgan_model
            ),
            is_valid_file=lambda p: p.endswith("bmp"),
        ),
        pin_memory=True,
        batch_size=8,
        num_workers=8,
        shuffle=True,
    )

    trainer = pl.Trainer(gpus=1, log_every_n_steps=1, callbacks=[eval_callback])
    trainer.fit(dt_model, dl)
