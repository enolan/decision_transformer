from pathlib import Path
import pytorch_lightning as pl
import torch
import torchvision


class FilteredImageFolder(torchvision.datasets.ImageFolder):
    """Dataset from a directory, filtering out empty directories."""

    def find_classes(self, directory):
        dirs, _mapping = super().find_classes(directory)
        out_dirs = []
        for dir in dirs:
            contents_iter = (Path(directory) / dir).iterdir()
            if any(True for _ in contents_iter):
                # ^ Weird Python method of checking if it has > 0 elements
                out_dirs.append(dir)
        out_mapping = {}
        next_idx = 0
        for out_dir in out_dirs:
            out_mapping[out_dir] = next_idx
            next_idx = next_idx + 1
        return out_dirs, out_mapping


class CheckGradients(pl.Callback):
    """Callback to log gradient norm to Tensorboard and optionally generate an
    SVG chart broken down by each weight."""

    def __init__(self, generate_charts=False, clip_at=None):
        super().__init__()
        self.generate_charts = generate_charts
        self.clip_at = clip_at

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        if pl_module.global_step % 100 == 0 and self.generate_charts:
            plot_grad_flow(pl_module.named_parameters())
            plt.savefig(f"grads-{pl_module.global_step}.svg")
            plt.clf()

        grad_norm, params = self._compute_grad_norm(pl_module.named_parameters())
        if self.clip_at is not None and grad_norm > self.clip_at:
            print(
                f"Gradient norm {grad_norm} at step {pl_module.global_step}, clipping"
            )
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.clip_at)
            new_norm = self._compute_grad_norm(pl_module.named_parameters())[0]
            print(f"New gradient norm is {new_norm}.")
            pl_module.log("grad_norm", new_norm, on_step=True)
        else:
            pl_module.log("grad_norm", grad_norm, on_step=True)

    def _compute_grad_norm(self, named_params):
        grads = []
        params = []
        for _n, param in named_params:
            if param.grad is not None:
                grads.append(param.grad.view(-1))
                params.append(param)
        return torch.linalg.vector_norm(torch.cat(grads)), params


# Debugging function from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and (p.grad is not None):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.figure(figsize=(20, 20))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    plt.tight_layout()


class TrainLossRecorder(pl.Callback):
    "Tool for recording the training loss."

    def __init__(self):
        self.train_loss = None

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.train_loss = outputs["loss"].item()
