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

import abc
import argparse
import subprocess
import sys

from pathlib import Path
from typing import Any, List

from compressai.datasets.rawvideo import get_raw_video_file_info


def run_command(cmd, ignore_returncodes=None):
    cmd = [str(c) for c in cmd]
    try:
        rv = subprocess.check_output(cmd)
        return rv.decode("ascii")
    except subprocess.CalledProcessError as err:
        if ignore_returncodes is not None and err.returncode in ignore_returncodes:
            return err.output
        print(err.output.decode("utf-8"))
        sys.exit(1)


def _get_ffmpeg_version():
    rv = run_command(["ffmpeg", "-version"])
    return rv.split()[2]


class Codec(abc.ABC):
    # name = ""
    description = ""
    help = ""

    @classmethod
    def setup_args(cls, parser):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError()

    @property
    def description(self):
        return self._description

    def add_parser_args(self, parser: argparse.ArgumentParser) -> None:
        pass

    def _set_args(self, args):
        return args

    @abc.abstractmethod
    def get_output_path(self, filepath: Path, **args: Any) -> Path:
        raise NotImplementedError

    @abc.abstractmethod
    def get_encode_cmd(self, filepath: Path, **args: Any) -> List[Any]:
        raise NotImplementedError


class x264(Codec):
    preset = ""
    tune = ""

    @property
    def name(self):
        return "x264"

    def description(self):
        return f"{self.name} {self.preset}, tuned for {self.tune}, ffmpeg version {_get_ffmpeg_version()}"

    def name_config(self):
        return f"{self.name}-{self.preset}-tune-{self.tune}"

    def add_parser_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("-p", "--preset", default="medium", help="preset")
        parser.add_argument(
            "--tune",
            default="psnr",
            help="tune encoder for psnr or ssim (default: %(default)s)",
        )

    def set_args(self, args):
        args = super()._set_args(args)
        self.preset = args.preset
        self.tune = args.tune
        return args

    def get_output_path(self, filepath: Path, qp, output: str) -> Path:
        return Path(output) / (
            f"{filepath.stem}_{self.name}_{self.preset}_tune-{self.tune}_qp{qp}.mp4"
        )

    def get_encode_cmd(self, filepath: Path, qp, outputdir) -> List[Any]:
        info = get_raw_video_file_info(filepath.stem)
        outputpath = self.get_output_path(filepath, qp, outputdir)
        cmd = [
            "ffmpeg",
            "-s:v",
            f"{info['width']}x{info['height']}",
            "-i",
            filepath,
            "-c:v",
            "h264",
            "-crf",
            qp,
            "-preset",
            self.preset,
            "-bf",
            0,
            "-tune",
            self.tune,
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "4",
            outputpath,
        ]
        return cmd


class x265(x264):
    @property
    def name(self):
        return "x265"

    def get_encode_cmd(self, filepath: Path, qp, outputdir) -> List[Any]:
        info = get_raw_video_file_info(filepath.stem)
        outputpath = self.get_output_path(filepath, qp, outputdir)
        cmd = [
            "ffmpeg",
            "-s:v",
            f"{info['width']}x{info['height']}",
            "-i",
            filepath,
            "-c:v",
            "hevc",
            "-crf",
            qp,
            "-preset",
            self.preset,
            "-x265-params",
            "bframes=0",
            "-tune",
            self.tune,
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "4",
            outputpath,
        ]
        return cmd
