from torch.nn.functional import F

def get_activation_fn(activation: str):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    raise ValueError(f"Activation should be relu/gelu, not {activation}.")