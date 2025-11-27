import fast_hadamard_transform
import torch


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return fast_hadamard_transform.hadamard_transform(x, scale)


def hadamard_transform_12n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.ops.sgl_kernel.fast_hadamard_transform_12N.default(x, scale)


def hadamard_transform_20n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.ops.sgl_kernel.fast_hadamard_transform_20N.default(x, scale)


def hadamard_transform_28n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.ops.sgl_kernel.fast_hadamard_transform_28N.default(x, scale)


def hadamard_transform_40n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.ops.sgl_kernel.fast_hadamard_transform_40N.default(x, scale)
