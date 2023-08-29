# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

import random


class ImageFolder_train(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

        self.scale_entry = []
        self.count = 0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")

        if self.count % 8 == 0:
            self.scale_entry = [0.25, random.uniform(0.25, 0.5), random.uniform(0.5, 1)]

        width_res2 = int(img.width * self.scale_entry[2])
        height_res2 = int(img.height * self.scale_entry[2])
        size_res2 = (width_res2, height_res2)

        width_res1 = int(img.width * self.scale_entry[1])
        height_res1 = int(img.height * self.scale_entry[1])
        size_res1 = (width_res1, height_res1)

        width_base = int(img.width * self.scale_entry[0])
        height_base = int(img.height * self.scale_entry[0])
        size_base = (width_base, height_base)

        img_res2 = self.transform(img.resize(size_res2))
        img_res1 = self.transform(img.resize(size_res1))
        img_base = self.transform(img.resize(size_base))

        self.count += 1
        return img_base, img_res1, img_res2

    def __len__(self):
        return len(self.samples)


class ImageFolder_test(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="test"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

        self.scale_entry = []

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")

        img_list = []
        scale_entry_list = []

        for i in range(0, 4):
            scale_2 = 1.0
            scale_1 = scale_2 / (1.25 + 0.25 * i)

            for ii in range(0, 4):
                scale_0 = scale_1 / (1.25 + 0.25 * ii)

                self.scale_entry = [scale_0, scale_1, scale_2]
                scale_entry_list.append(self.scale_entry)

                width_res2 = int(img.width * self.scale_entry[2])
                height_res2 = int(img.height * self.scale_entry[2])
                size_res2 = (width_res2, height_res2)

                width_res1 = int(img.width * self.scale_entry[1])
                height_res1 = int(img.height * self.scale_entry[1])
                size_res1 = (width_res1, height_res1)

                width_base = int(img.width * self.scale_entry[0])
                height_base = int(img.height * self.scale_entry[0])
                size_base = (width_base, height_base)

                img_res2 = self.transform(img.resize(size_res2))
                img_res1 = self.transform(img.resize(size_res1))
                img_base = self.transform(img.resize(size_base))

                img_list.append([img_base, img_res1, img_res2])

        return img_list

    def __len__(self):
        return len(self.samples)
