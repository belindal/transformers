# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert Linformer checkpoint."""


import argparse
import logging
import pathlib

import fairseq
import torch
from fairseq.models.fb_linformer import LinformerModel as FairseqLinformerModel
from fairseq.modules.fb_linformer_sentence_encoder_layer import LinformerSentenceEncoderLayer
from packaging import version

from transformers.modeling_bert import BertIntermediate, BertLayer, BertOutput, BertSelfAttention, BertSelfOutput
from src.transformers.modeling_linformer import LinformerConfig, LinformerForMaskedLM, LinformerForSequenceClassification


if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_TEXT = "Hello world! cécé herlolip"


def convert_linformer_checkpoint_to_pytorch(
    linformer_checkpoint_path: str, pytorch_dump_folder_path: str, classification_head: bool
):
    """
    Copy/paste/tweak linformer's weights to our BERT structure.
    """
    linformer = FairseqLinformerModel.from_pretrained(linformer_checkpoint_path)
    linformer.eval()  # disable dropout
    linformer_sent_encoder = linformer.model.encoder.sentence_encoder
    config = LinformerConfig(
        vocab_size=linformer_sent_encoder.embed_tokens.num_embeddings,
        hidden_size=linformer.args.encoder_embed_dim,
        num_hidden_layers=linformer.args.encoder_layers,
        num_attention_heads=linformer.args.encoder_attention_heads,
        intermediate_size=linformer.args.encoder_ffn_embed_dim,
        max_position_embeddings=linformer_sent_encoder.embed_positions.num_embeddings,
        type_vocab_size=1,
        layer_norm_eps=1e-5,  # PyTorch default used in fairseq
        max_seq_len=linformer.args.max_positions,
        compressed=linformer.args.compressed,
        shared_kv_compressed=(linformer.args.shared_kv_compressed == 1),
        shared_layer_kv_compressed=(linformer.args.shared_layer_kv_compressed == 1),
        freeze_compress=(linformer.args.freeze_compress == 1),
    )
    if classification_head:
        config.num_labels = linformer.model.classification_heads["mnli"].out_proj.weight.shape[0]
    print("Our BERT config:", config)

    model = LinformerForSequenceClassification(config) if classification_head else LinformerForMaskedLM(config)
    model.eval()

    # Now let's copy all the weights.
    fairseq_linformer_state = linformer_sent_encoder.state_dict()
    # Embeddings
    model.roberta.embeddings.word_embeddings.weight.data = fairseq_linformer_state.pop('embed_tokens.weight')
    model.roberta.embeddings.position_embeddings.weight.data = fairseq_linformer_state.pop('embed_positions.weight')
    model.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.roberta.embeddings.token_type_embeddings.weight
    )  # just zero them out b/c Linformer doesn't use them.
    model.roberta.embeddings.LayerNorm.weight.data = fairseq_linformer_state.pop('emb_layer_norm.weight')
    model.roberta.embeddings.LayerNorm.bias.data = fairseq_linformer_state.pop('emb_layer_norm.bias')
    
    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        layer: BertLayer = model.roberta.encoder.layer[i]
        linformer_layer: LinformerSentenceEncoderLayer = linformer_sent_encoder.layers[i]

        # self attention
        self_attn: BertSelfAttention = layer.attention.self
        assert (
            linformer_layer.self_attn.k_proj.weight.data.shape
            == linformer_layer.self_attn.q_proj.weight.data.shape
            == linformer_layer.self_attn.v_proj.weight.data.shape
            == torch.Size((config.hidden_size, config.hidden_size))
        )

        self_attn.query.weight.data = fairseq_linformer_state.pop('layers.{0}.self_attn.q_proj.weight'.format(i))
        self_attn.query.bias.data = fairseq_linformer_state.pop('layers.{0}.self_attn.q_proj.bias'.format(i))
        self_attn.key.weight.data = fairseq_linformer_state.pop('layers.{0}.self_attn.k_proj.weight'.format(i))
        self_attn.key.bias.data = fairseq_linformer_state.pop('layers.{0}.self_attn.k_proj.bias'.format(i))
        self_attn.value.weight.data = fairseq_linformer_state.pop('layers.{0}.self_attn.v_proj.weight'.format(i))
        self_attn.value.bias.data = fairseq_linformer_state.pop('layers.{0}.self_attn.v_proj.bias'.format(i))

        # linformer compression
        assert self_attn.shared_kv_compressed == layer.attention.self.shared_kv_compressed
        assert linformer_layer.self_attn.compress_k.weight.data.shape == torch.Size((
            linformer.args.max_positions // linformer.args.compressed, linformer.args.max_positions,
        )) == layer.attention.self.compress_k.weight.data.shape
        self_attn.compress_k.weight.data = fairseq_linformer_state.pop('layers.{0}.self_attn.compress_k.weight'.format(i))
        self_attn.compress_k.bias.data = fairseq_linformer_state.pop('layers.{0}.self_attn.compress_k.bias'.format(i))
        if not self_attn.shared_kv_compressed:
            assert (
                linformer_layer.self_attn.compress_v.weight.data.shape
                == self_attn.compress_v.weight.data.shape
            )
            self_attn.compress_v.weight.data = fairseq_linformer_state.pop('layers.{0}.self_attn.compress_v.weight'.format(i))
        if layer.shared_compress_layer is not None:
            layer.shared_compress_layer.weight.data = fairseq_linformer_state.pop('layers.{0}.shared_compress_layer.weight'.format(i))
            layer.shared_compress_layer.bias.data = fairseq_linformer_state.pop('layers.{0}.shared_compress_layer.bias'.format(i))

        # self-attention output
        self_output: BertSelfOutput = layer.attention.output
        assert self_output.dense.weight.shape == linformer_layer.self_attn.out_proj.weight.shape
        self_output.dense.weight.data = fairseq_linformer_state.pop('layers.{0}.self_attn.out_proj.weight'.format(i))
        self_output.dense.bias.data = fairseq_linformer_state.pop('layers.{0}.self_attn.out_proj.bias'.format(i))
        self_output.LayerNorm.weight.data = fairseq_linformer_state.pop('layers.{0}.self_attn_layer_norm.weight'.format(i))
        self_output.LayerNorm.bias.data = fairseq_linformer_state.pop('layers.{0}.self_attn_layer_norm.bias'.format(i))

        # intermediate
        intermediate: BertIntermediate = layer.intermediate
        assert intermediate.dense.weight.shape == linformer_layer.fc1.weight.shape
        intermediate.dense.weight.data = fairseq_linformer_state.pop('layers.{0}.fc1.weight'.format(i))
        intermediate.dense.bias.data = fairseq_linformer_state.pop('layers.{0}.fc1.bias'.format(i))

        # output
        bert_output: BertOutput = layer.output
        assert bert_output.dense.weight.shape == linformer_layer.fc2.weight.shape
        bert_output.dense.weight.data = fairseq_linformer_state.pop('layers.{0}.fc2.weight'.format(i))
        bert_output.dense.bias.data = fairseq_linformer_state.pop('layers.{0}.fc2.bias'.format(i))
        bert_output.LayerNorm.weight.data = fairseq_linformer_state.pop('layers.{0}.final_layer_norm.weight'.format(i))
        bert_output.LayerNorm.bias.data = fairseq_linformer_state.pop('layers.{0}.final_layer_norm.bias'.format(i))
        # end of layer

    # shared compression between layers 
    if config.shared_layer_kv_compressed:
        model.roberta.encoder.compress_layer.weight.data = fairseq_linformer_state.pop('compress_layer.weight'.format(i))
        model.roberta.encoder.compress_layer.bias.data = fairseq_linformer_state.pop('compress_layer.bias'.format(i))

    assert len(fairseq_linformer_state) == 0

    if classification_head:
        model.classifier.dense.weight = linformer.model.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = linformer.model.classification_heads["mnli"].dense.bias
        model.classifier.out_proj.weight = linformer.model.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = linformer.model.classification_heads["mnli"].out_proj.bias
    else:
        # LM Head
        model.lm_head.dense.weight = linformer.model.encoder.lm_head.dense.weight
        model.lm_head.dense.bias = linformer.model.encoder.lm_head.dense.bias
        model.lm_head.layer_norm.weight = linformer.model.encoder.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = linformer.model.encoder.lm_head.layer_norm.bias
        model.lm_head.decoder.weight = linformer.model.encoder.lm_head.weight
        model.lm_head.decoder.bias = linformer.model.encoder.lm_head.bias

    # Let's check that we get the same results.
    input_ids: torch.Tensor = linformer.encode(SAMPLE_TEXT).unsqueeze(0)  # batch of size 1

    our_output = model(input_ids)[0]
    if classification_head:
        their_output = linformer.model.classification_heads["mnli"](linformer.extract_features(input_ids))
    else:
        their_output = linformer.model(input_ids)[0]
    print(our_output.shape, their_output.shape)
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    success = torch.allclose(our_output, their_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")

    pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--linformer_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    args = parser.parse_args()
    convert_linformer_checkpoint_to_pytorch(
        args.linformer_checkpoint_path, args.pytorch_dump_folder_path, args.classification_head
    )
