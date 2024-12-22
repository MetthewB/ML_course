import random
import torch
from torchvision import datasets

class ModifiedDataset(datasets.MNIST):
    def __init__(
        self,
        root,
        img_size=56,
        random_shift=False,
        scramble_image=False,
        noise=0.0,
        *args,
        **kwargs
    ):
        super().__init__(root, *args, **kwargs)
        assert img_size >= 28
        self.img_size = img_size
        self.scramble_image = scramble_image
        assert noise >= 0.0
        self.noise = noise

        if random_shift:
            rng = random.Random(433)
            self.r_idxs = [
                rng.randrange(img_size - 28 + 1) for _ in range(len(self))
            ]
            self.c_idxs = [
                rng.randrange(img_size - 28 + 1) for _ in range(len(self))
            ]
        else:
            self.r_idxs = [(img_size - 28) // 2] * len(self)
            self.c_idxs = self.r_idxs
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(2147483647)
        self.shuffle_idxs = torch.randperm(img_size**2, generator=self.torch_rng)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        image, label = sample

        if self.img_size > 28:
            new_image = torch.full((1, self.img_size, self.img_size), image.min())
            c_idx = self.c_idxs[index]
            r_idx = self.r_idxs[index]
            new_image[:, c_idx : c_idx + 28, r_idx : r_idx + 28] = image
            image = new_image

        if self.noise:
            self.torch_rng.manual_seed(2147433433 + index)
            image = image + self.noise * torch.randn(
                image.shape, generator=self.torch_rng
            )

        if self.scramble_image:
            image = image.view(-1)[self.shuffle_idxs].reshape(
                1, self.img_size, self.img_size
            )

        return (image, label)