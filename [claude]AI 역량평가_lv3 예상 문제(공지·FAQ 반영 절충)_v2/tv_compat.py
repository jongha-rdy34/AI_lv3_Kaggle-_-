"""torchvision이 없는 PC에서만 쓰는 연습용 대체 모듈.

시험 환경에는 torchvision이 설치되어 있을 가능성이 높습니다. 노트북은 먼저
`from torchvision.transforms import v2 as T`를 시도하고, 실패할 때만 이 모듈을 씁니다.
이름·인자·동작을 torchvision.transforms.v2의 자주 쓰는 부분과 같게 맞췄으므로
여기서 연습한 코드를 시험에서 그대로 쓸 수 있습니다. (공식 구현이 아닙니다.)

지원: Compose, ToImage, ToDtype, RandomHorizontalFlip, RandomVerticalFlip,
      ColorJitter(brightness, contrast), Normalize, RandomApply, Lambda
"""
import random

import numpy as np
import torch


class _Namespace:
    pass


def _as_tensor_image(img):
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = img[..., None]
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)  # HWC → CHW
    return img


class Compose:
    """여러 변환을 순서대로 적용합니다. Compose([t1, t2, ...])"""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        inner = "\n".join(f"    {t!r}" for t in self.transforms)
        return f"Compose(\n{inner}\n)"


class ToImage:
    """numpy (H,W,C) 배열 또는 텐서 → (C,H,W) 텐서. 자료형은 바꾸지 않습니다(uint8 그대로)."""

    def __call__(self, img):
        return _as_tensor_image(img)

    def __repr__(self):
        return "ToImage()"


class ToDtype:
    """자료형 변환. scale=True이고 정수→실수이면 [0, 255] → [0, 1]로 나눕니다."""

    def __init__(self, dtype, scale=False):
        self.dtype, self.scale = dtype, scale

    def __call__(self, img):
        img = _as_tensor_image(img)
        if self.scale and not img.is_floating_point() and self.dtype.is_floating_point:
            return img.to(self.dtype) / 255.0
        return img.to(self.dtype)

    def __repr__(self):
        return f"ToDtype(dtype={self.dtype}, scale={self.scale})"


class RandomHorizontalFlip:
    """확률 p로 좌우 반전(마지막 축)."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        img = _as_tensor_image(img)
        return img.flip(-1) if torch.rand(1).item() < self.p else img

    def __repr__(self):
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip:
    """확률 p로 상하 반전(끝에서 두 번째 축)."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        img = _as_tensor_image(img)
        return img.flip(-2) if torch.rand(1).item() < self.p else img

    def __repr__(self):
        return f"RandomVerticalFlip(p={self.p})"


def _range(v, name):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if v < 0:
            raise ValueError(f"{name}은 0 이상이어야 합니다.")
        return (max(0.0, 1 - v), 1 + v)
    lo, hi = v
    return (float(lo), float(hi))


class ColorJitter:
    """밝기·대비를 무작위로 바꿉니다(이 대체 모듈은 saturation·hue를 지원하지 않음).

    brightness: float b → 배율 U[max(0,1-b), 1+b], 또는 (min, max) 튜플.
    실수 텐서는 [0,1], uint8 텐서는 [0,255]로 자릅니다.
    """

    def __init__(self, brightness=None, contrast=None, saturation=None, hue=None):
        if saturation is not None or hue is not None:
            raise NotImplementedError("tv_compat은 saturation·hue를 지원하지 않습니다(torchvision에서는 지원).")
        self.brightness, self.contrast = _range(brightness, "brightness"), _range(contrast, "contrast")

    def __call__(self, img):
        img = _as_tensor_image(img)
        is_float = img.is_floating_point()
        x = img.float() if not is_float else img
        top = 1.0 if is_float else 255.0
        order = [k for k in ("brightness", "contrast") if getattr(self, k) is not None]
        random.shuffle(order)
        for k in order:
            lo, hi = getattr(self, k)
            f = float(torch.empty(1).uniform_(lo, hi))
            if k == "brightness":
                x = (x * f).clamp(0, top)
            else:
                gray_mean = x.mean()
                x = ((x - gray_mean) * f + gray_mean).clamp(0, top)
        return x if is_float else x.round().to(img.dtype)

    def __repr__(self):
        return f"ColorJitter(brightness={self.brightness}, contrast={self.contrast})"


class Normalize:
    """채널별 (x - mean) / std. 입력은 실수 텐서 (C,H,W) 또는 (B,C,H,W)."""

    def __init__(self, mean, std):
        self.mean, self.std = list(mean), list(std)

    def __call__(self, img):
        if not img.is_floating_point():
            raise TypeError("Normalize는 실수 텐서에만 적용합니다. 먼저 ToDtype(torch.float32, scale=True)를 쓰세요.")
        shape = (-1, 1, 1)
        m = torch.tensor(self.mean, dtype=img.dtype).view(shape)
        s = torch.tensor(self.std, dtype=img.dtype).view(shape)
        return (img - m) / s

    def __repr__(self):
        return f"Normalize(mean={self.mean}, std={self.std})"


class RandomApply:
    """확률 p로 transforms 목록 전체를 적용합니다."""

    def __init__(self, transforms, p=0.5):
        self.transforms, self.p = list(transforms), p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            for t in self.transforms:
                img = t(img)
        return img


class Lambda:
    """임의의 함수를 변환으로 씁니다."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, img):
        return self.fn(img)


transforms_v2 = _Namespace()
for _name in ["Compose", "ToImage", "ToDtype", "RandomHorizontalFlip", "RandomVerticalFlip",
              "ColorJitter", "Normalize", "RandomApply", "Lambda"]:
    setattr(transforms_v2, _name, globals()[_name])
