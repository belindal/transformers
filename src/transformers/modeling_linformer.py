# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Linformer model. """

import logging
import math
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from .configuration_linformer import LinformerConfig
from .file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from .modeling_bert import (
    BertAttention,
    BertEncoder,
    BertIntermediate,
    BertLayer,
    BertLayerNorm,
    BertOutput,
    BertPooler,
    BertPreTrainedModel,
    BertSelfAttention,
    BertSelfOutput,
)
from .modeling_roberta import (
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
)
from .modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from .modeling_roberta import RobertaEmbeddings, RobertaLMHead
from .modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer


logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "LinformerConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

LINFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = []


class LinformerSelfAttention(BertSelfAttention):
    def __init__(self, config, shared_compress_layer=None):
        super().__init__(config)

        # used for compressing sequence to subsequence
        if shared_compress_layer is None:
            self.compress_seq_len = config.max_position_embeddings // config.compressed
            self.compress_k = nn.Linear(config.max_position_embeddings, self.compress_seq_len, bias=False)
            nn.init.xavier_uniform_(self.compress_k.weight, gain=1/math.sqrt(2))
            if not config.shared_kv_compressed:
                self.compress_v = nn.Linear(config.max_position_embeddings, self.compress_seq_len, bias=False)
                nn.init.xavier_uniform_(self.compress_v.weight, gain=1/math.sqrt(2))
        else:
            self.compress_k = shared_compress_layer
            if not config.shared_kv_compressed:
                # TODO this still shares the layer between key/value...
                self.compress_v = shared_compress_layer
    
        self.shared_kv_compressed = config.shared_kv_compressed
    
        if config.freeze_compress:
            self.compress_k.weight.requires_grad = False
            if not config.shared_kv_compressed:
                self.compress_v.weight.requires_grad = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        did_compress = False

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            # mixed_key_layer = self.key(hidden_states)
            # mixed_value_layer = self.value(hidden_states)
            tgt_len = hidden_states.size(1)
            # (bsz, embed_dim, seqlen)
            k_input = hidden_states.permute(0, 2, 1).contiguous()
            k_input = F.linear(k_input, self.compress_k.weight[:, 0: tgt_len]).permute(0, 2, 1).contiguous()
            mixed_key_layer = self.key(k_input)
            # (bsz, embed_dim, seqlen)
            v_input = hidden_states.permute(0, 2, 1).contiguous()
            if self.shared_kv_compressed:  # use shared kv compressed linear layer
                v_input = F.linear(v_input, self.compress_k.weight[:, 0: tgt_len]).permute(0, 2, 1).contiguous()
            else:
                v_input = F.linear(v_input, self.compress_v.weight[:, 0: tgt_len]).permute(0, 2, 1).contiguous()
            mixed_value_layer = self.value(v_input)
            did_compress = True

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None and not did_compress:
            # don't mask if we've already compressed (all values are linear combination of all other values)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class LinformerAttention(BertAttention):
    def __init__(self, config, shared_compress_layer=None):
        super().__init__(config)
        self.self = LinformerSelfAttention(config, shared_compress_layer)


class LinformerLayer(BertLayer):
    def __init__(self, config, shared_compress_layer=None):
        super().__init__(config)
        if not self.is_decoder:
            self.shared_compress_layer = shared_compress_layer
            self.attention = LinformerAttention(config, shared_compress_layer)


class LinformerEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)

        if config.shared_layer_kv_compressed:
            compress_layer = nn.Linear(config.max_position_embeddings, config.max_position_embeddings // config.compressed)
            # intialize parameters for compressed layer
            nn.init.xavier_uniform_(compress_layer.weight, gain=1 / math.sqrt(2))
            if config.freeze_compress:
                compress_layer.weight.requires_grad = False
            self.compress_layer = compress_layer

        self.layer = nn.ModuleList([LinformerLayer(
            config, shared_compress_layer=(self.compress_layer if config.shared_layer_kv_compressed else None),
        ) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
	# TODO zero out padding?
        return super().forward(
	    hidden_states=hidden_states,
            attention_mask=attention_mask,
	    head_mask=head_mask,
	    encoder_hidden_states=encoder_hidden_states,
	    encoder_attention_mask=encoder_attention_mask,
	    output_attentions=output_attentions,
	    output_hidden_states=output_hidden_states,
	    return_dict=return_dict,
        )


class LinformerPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained
        models.
    """
    # TODO note this classes is not actually inherited as we inherit from robertamodel -> bertpretrainedmodel
    # as of now, code is identical to bertpretrainedmodel, so this is okay

    config_class = LinformerConfig
    base_model_prefix = "roberta"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            # doesn't override the weights in compress_layer that we initialized with xavier
            module.bias.data.zero_()


class LinformerModel(RobertaModel):

    config_class = LinformerConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.encoder = LinformerEncoder(config)
        self.init_weights()


class LinformerForMaskedLM(RobertaForMaskedLM):
    config_class = LinformerConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = LinformerModel(config)
        self.init_weights()


class LinformerForSequenceClassification(RobertaForSequenceClassification):
    config_class = LinformerConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = LinformerModel(config)
        self.init_weights()


class LinformerForQuestionAnswering(RobertaForQuestionAnswering):
    config_class = LinformerConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = LinformerModel(config)
        self.init_weights()


class LinformerForTokenClassification(RobertaForTokenClassification):
    config_class = LinformerConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = LinformerModel(config)
        self.init_weights()


class LinformerForMultipleChoice(RobertaForMultipleChoice):
    config_class = LinformerConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = LinformerModel(config)
        self.init_weights()
