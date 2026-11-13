"""Recover and evaluate frozen two-dimensional random Fourier position grids."""

import math

import torch


def fourier_grid(size, frequencies):
    height, width = size
    if min(height, width) < 1 or frequencies.ndim != 2 or frequencies.shape[0] != 2:
        raise ValueError('Invalid Fourier position grid dimensions')
    y = (torch.arange(height, device=frequencies.device, dtype=frequencies.dtype) + .5) / height
    x = (torch.arange(width, device=frequencies.device, dtype=frequencies.dtype) + .5) / width
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    phase = ((2 * torch.stack((xx, yy), dim=-1) - 1) @ frequencies) * (2 * math.pi)
    return torch.cat((phase.sin(), phase.cos()), dim=-1).permute(2, 0, 1).unsqueeze(0)


def recover_frequencies(embedding):
    if (embedding.ndim != 4 or embedding.shape[0] != 1 or embedding.shape[1] % 2
            or min(embedding.shape[-2:]) < 2 or not embedding.is_floating_point()):
        raise ValueError('Expected a single sine/cosine position grid with two spatial axes')
    if not torch.isfinite(embedding).all():
        raise ValueError('Nonfinite Fourier position grid')
    sine, cosine = embedding.detach().double().chunk(2, dim=1)
    frequencies = []
    for axis in (-1, -2):
        before = [slice(None)] * 4
        after = [slice(None)] * 4
        before[axis], after[axis] = slice(None, -1), slice(1, None)
        sa, ca = sine[tuple(after)], cosine[tuple(after)]
        sb, cb = sine[tuple(before)], cosine[tuple(before)]
        difference_sine = (sa * cb - ca * sb).sum(dim=(0, 2, 3))
        difference_cosine = (ca * cb + sa * sb).sum(dim=(0, 2, 3))
        frequencies.append(torch.atan2(difference_sine, difference_cosine)
                           * embedding.shape[axis] / (4 * math.pi))
    result = torch.stack(frequencies)
    recovered = fourier_grid(embedding.shape[-2:], result)
    tolerance = max(5e-5, 2 * torch.finfo(embedding.dtype).eps)
    if not torch.allclose(recovered, embedding.double(), atol=tolerance, rtol=0):
        raise ValueError('Position grid does not encode a consistent unaliased Fourier basis')
    return result


def resize_fourier_grid(embedding, size):
    if tuple(size) == tuple(embedding.shape[-2:]):
        return embedding
    return fourier_grid(size, recover_frequencies(embedding)).to(embedding.dtype)
