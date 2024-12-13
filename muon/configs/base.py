# Copyright 2024 Lucas Dax Lingle
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from ml_collections import config_dict


def get_base_config():
    config = config_dict.ConfigDict()

    config.n_mesh_rows = 128
    config.n_mesh_cols = 1

    # logging/plotting
    config.sow_intermediates = False
    config.sow_param_info = False
    config.is_sweep = False

    # huggingface tokenizer and dataset settings
    config.hftr_tokenizer_name = "T5TokenizerFast"
    config.hftr_tokenizer_instance = "t5-base"
    config.hfds_identifier = "c4"
    config.hfds_config = "en"
    config.hfds_datacol = "text"
    config.hfds_buffer_size = 512  # example buffer length for batched tokenization
    config.sequence_len = 256
    config.force_download = True  # should be true unless you know what you're doing
    config.n_ds_shard = 0  # 0 = shard by host; less < n_host = subshard existing shards

    # architecture
    config.param_dtype = "float32"  # master copy of weights in fp32
    config.dtype = "bfloat16"  # weights and activations are in bfloat16 on fwd/bwd
    config.n_layer = 8  # number of transformer blocks
    config.d_model = 1024  # residual stream width
    config.d_head = 128  # attention head width
    config.n_head = 8  # number of query heads
    config.n_kv_head = 1  # number of key/value heads; equal to the number of GQA groups
    config.rope_theta = 10_000
    config.ff_mult = 4.0  # positionwise ff hidden width: d_ff = ff_mult * d_model
    config.ff_act = "sqrelu"  # any activation in jax.nn, or "swiglu", or "sqrelu".
    config.rmsnorm_eps = 1e-5  # rmsnorm epsilon
    config.rmsnorm_params = False  # rmsnorm trainable

    # optimization
    config.bsz = 2**22  # global batch size, in tokens
    config.gc = 1.0  # grad clip
    config.lr = 0.25  # learning rate
    config.sched = "cosine"
    config.optim = "spectral_muon"  # one of adam, lion, muon, spectral_muon
    config.wd = 0.0001  # indep weight decay lambda

    # periodic action settings
    config.n_print_step = 100  # print every
    config.n_save_step = 5000  # checkpoint every
    config.n_eval_step = 100  # eval steps per checkpoint
    config.n_warmup_step = 10_000  # warmup steps during pretraining
    config.n_pretrain_step = 125_000  # pretraining steps
    config.n_finetune_step = 0  # finetuning steps, keep zero during pretraining
    config.no_checkpoint = False  # skip saving the model

    # sampling settings
    config.sampling_method = "nucleus"
    config.sampling_nucleus = 0.8
    config.sampling_prompt_len = 128
    config.sampling_max_len = 1024

    return config


def get_config():
    return get_base_config()
