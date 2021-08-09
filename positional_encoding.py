import math
import torch


def positional_encoding(d_model, n_tokens):
    "Generate a positional encoding for a transformer"
    token_idxs = torch.arange(n_tokens).unsqueeze(1)
    if d_model % 2 == 0:
        d_model_rounded = d_model
    else:
        d_model_rounded = d_model + 1
    div_term = torch.exp(
        torch.arange(0, d_model_rounded, 2) * (-math.log(10000.0) / d_model)
    )
    pos_e = torch.zeros(n_tokens, d_model)
    # If there's an odd number of d_model we treat that specially
    pos_e[:, 0::2] = torch.sin(token_idxs * div_term)
    if d_model % 2 == 0:
        pos_e[:, 1::2] = torch.cos(token_idxs * div_term)
    else:
        pos_e[:, 1::2] = torch.cos(token_idxs * div_term[:-1])
    return pos_e
