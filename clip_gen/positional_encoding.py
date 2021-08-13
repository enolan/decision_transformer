import math
import torch


def gen_positional_encoding(d_model, n_tokens):
    "Generate a positional encoding for a transformer"
    pos_e = torch.zeros(n_tokens, d_model)
    for pos in range(n_tokens):
        for i in range(0, d_model, 2):
            divisor = 10000 ** (i / d_model)
            pos_e[pos, i] = math.sin(pos / divisor)
            pos_e[pos, i + 1] = math.cos(pos / divisor)
        if d_model % 2 != 0:
            pos_e[pos, d_model - 1] = math.sin(pos / 10000 ** ((d_model - 1) / d_model))
    return pos_e
