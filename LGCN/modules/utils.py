import torch


def penalty_builder(penalty_config):
    """Build length penalty for beam search.

    Supported formats:
      - '' / None / 0 / '0': disabled
      - 'wu_1.0': Wu et al. length penalty with alpha=1.0
      - 'avg_1.0': average-length penalty
      - '1.0': shorthand for 'wu_1.0'
    """
    if penalty_config is None:
        return lambda x, y: y
    penalty_config = str(penalty_config).strip()
    if penalty_config == '' or penalty_config in {'0', '0.0', 'none', 'None'}:
        return lambda x, y: y

    if '_' not in penalty_config:
        pen_type, alpha = 'wu', float(penalty_config)
    else:
        pen_type, alpha = penalty_config.split('_', 1)
        alpha = float(alpha)

    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)
    raise ValueError(f"Unknown length penalty config: {penalty_config}")



def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x
