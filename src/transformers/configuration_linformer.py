# coding=utf_8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE_2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Linformer configuration """

import logging
from typing import List, Union

from .configuration_roberta import RobertaConfig


logger = logging.getLogger(__name__)

LINFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class LinformerConfig(RobertaConfig):
    model_type = "linformer"

    def __init__(
        self,
        compressed: int = 1,
        shared_kv_compressed: bool = False,
        shared_layer_kv_compressed: bool = False,
        freeze_compress: bool = False,
        **kwargs,
    ):
        """
        compressed: compressed ratio of sequence length
            (TODO should this be a float??)
        shared_kv_compressed: share compressed matrix between k and v, in each layer
        shared_layer_kv_compressed: share compressed matrix between k and v and across all layers
        freeze_compress: freeze the parameters in compressed layer
        """
        super().__init__(**kwargs)
        self.compressed = compressed
        self.shared_kv_compressed = shared_kv_compressed
        self.shared_layer_kv_compressed = shared_layer_kv_compressed
        self.freeze_compress = freeze_compress
