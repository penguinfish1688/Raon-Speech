# Copyright 2026 The Qwen team, Alibaba Group, the HuggingFace Inc. team, and The RAON Authors. All rights reserved.
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
#
# Built on top of Qwen3 and Qwen3OmniMoe models from HuggingFace Transformers.

"""Core raon model definition with training forward pass, loss computation, and audio codec integration."""

from __future__ import annotations

# Loss methods (unreduced_causal_lm_loss, _compute_audio_loss, _combine_losses, etc.) live in RaonLossMixin (loss.py).
import logging
import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Self, cast

import torch
import torch.nn.functional as F
import torchaudio.functional
from torch import nn
from transformers import (
    AutoTokenizer,
    Cache,
    DynamicCache,
    MimiConfig,
    PretrainedConfig,
    PreTrainedModel,
    Qwen3Config,
    Qwen3Model,
    StaticCache,
)
from transformers.models.mimi.modeling_mimi import MimiEncoderOutput
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeTalkerCodePredictorConfig,
)
from transformers.utils.generic import ModelOutput

from ..modules import (
    EmbeddingAdaptor,
    EmbeddingAdaptorConfig,
    EmbeddingAdaptorOutput,
    PretrainedSpeakerEncoder,
    RaonCodePredictorModel,
    SpeakerEncoderConfig,
    ThinkerToTalkerProjection,
)
from ..modules.audio_encoder import AuTWrapper
from ..modules.audio_tokenizer import (
    CausalAudioEncoder,
    CausalAudioEncoderOutput,
    MimiConv1dPaddingCache,
    StreamingMimiDecoderOutput,
    StreamingMimiModel,
)
from ..modules.voxtral_encoder import VoxtralRealtimeEncoderConfig
from ..modules.voxtral_wrapper import VoxtralWrapper
from ..types import (
    AudioDecoderOutput,
    AudioEncoderOutput,
    AudioTokenizerOutput,
)
from ..utils.loss import RaonLossMixin
from ..utils.misc import (
    _read_acoustic_loss_weights,
    _read_loss_param,
    cast_float_inputs,
    cast_to_module_dtype,
)
from ..utils.special_tokens import (
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_BC,
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_START,
    DUPLEX_SIL,
    IM_START,
    LOSS_IGNORE_INDEX,
    SPEAKER_EMBEDDING_PLACEHOLDER,
)
from .wrapper import RaonInferenceModel

TEXT_MODEL_CONFIGS: dict[str, type[PretrainedConfig]] = {
    Qwen3Config.model_type: Qwen3Config,
}
TEXT_MODELS: dict[str, type[PreTrainedModel]] = {
    Qwen3Config.model_type: Qwen3Model,
}

logger = logging.getLogger(__name__)


class RaonConfig(PretrainedConfig):
    """Configuration class for RaonModel."""

    model_type = "raon"
    has_no_defaults_at_init = True
    text_model_config: PretrainedConfig = None
    audio_encoder_config: Qwen3OmniMoeAudioEncoderConfig | VoxtralRealtimeEncoderConfig = None
    audio_tokenizer_config: MimiConfig = None
    input_adaptor_config: EmbeddingAdaptorConfig = None
    output_adaptor_config: EmbeddingAdaptorConfig = None
    code_predictor_config: Qwen3OmniMoeTalkerCodePredictorConfig = None
    speaker_encoder_config: SpeakerEncoderConfig | None = None
    # Note: speaker_encoder_config is intentionally excluded from sub_configs.
    # It is optional (can be None), and transformers' _get_dtype unconditionally
    # calls sub_config.dtype on every entry, which crashes on None.
    # Deserialization from dict is handled in __init__ instead.
    sub_configs = {
        "text_model_config": PretrainedConfig,
        "audio_encoder_config": PretrainedConfig,
        "audio_tokenizer_config": PretrainedConfig,
        "input_adaptor_config": EmbeddingAdaptorConfig,
        "output_adaptor_config": EmbeddingAdaptorConfig,
        "code_predictor_config": Qwen3OmniMoeTalkerCodePredictorConfig,
    }

    def __init__(
        self,
        *,
        text_model_config: dict[str, Any] | PretrainedConfig | None = None,
        audio_encoder_config: dict[str, Any] | Qwen3OmniMoeAudioEncoderConfig | VoxtralRealtimeEncoderConfig | None = None,
        audio_tokenizer_config: dict[str, Any] | MimiConfig | None = None,
        input_adaptor_config: dict[str, Any] | EmbeddingAdaptorConfig | None = None,
        output_adaptor_config: dict[str, Any] | EmbeddingAdaptorConfig | None = None,
        code_predictor_config: dict[str, Any] | Qwen3OmniMoeTalkerCodePredictorConfig | None = None,
        speaker_encoder_config: dict[str, Any] | SpeakerEncoderConfig | None = None,
        num_talker_layers: int = 0,
        supports_audio_input: bool = True,
        supports_audio_output: bool = True,
        aut_is_causal: bool = False,
        proj_code_bias: bool = False,
        accept_hidden_layer: int = -1,
        talker_config: dict[str, Any] | PretrainedConfig | None = None,
        thinker_to_talker_pre_norm: bool = False,
        sequence_mode: str | None = None,
        use_sil_token: bool = False,
        no_audio_in_sil: bool = False,
        text_lookahead: int = 0,
        use_duplex_end_pad: bool = False,
        speaker_embedding_to_code_predictor: bool = True,
        duplex_pad_token_id: int | None = None,
        duplex_end_pad_token_id: int | None = None,
        duplex_sil_token_id: int | None = None,
        duplex_bc_token_id: int | None = None,
        use_backchannel_token: bool = False,
        bc_loss_weight: float = 1.0,
        speaker_token_id: int | None = None,
        audio_input_token_id: int | None = None,
        audio_output_token_id: int | None = None,
        audio_start_token_id: int | None = None,
        im_start_token_id: int | None = None,
        text_loss_weight: float = 1.0,
        sil_loss_weight: float = 1.0,
        epad_loss_weight: float = 0.0,
        semantic_loss_weight: float = 1.0,
        acoustic_loss_weights: list[float] | None = None,
        audio_lm_head_enabled: bool = True,
        delays: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Ensure auto_map is always serialized for trust_remote_code Hub loading.
        if not hasattr(self, "auto_map") or not self.auto_map:
            self.auto_map = {
                "AutoConfig": "raon--models--raon.RaonConfig",
                "AutoModel": "raon--models--raon.RaonModel",
            }

        assert text_model_config is not None, "RaonConfig: `text_model_config` is required."
        assert audio_encoder_config is not None, "RaonConfig: `audio_encoder_config` is required."
        assert audio_tokenizer_config is not None, "RaonConfig: `audio_tokenizer_config` is required."
        assert input_adaptor_config is not None, "RaonConfig: `input_adaptor_config` is required."
        assert output_adaptor_config is not None, "RaonConfig: `output_adaptor_config` is required."
        assert code_predictor_config is not None, "RaonConfig: `code_predictor_config` is required."

        if isinstance(text_model_config, dict):
            model_type = text_model_config.get("model_type", Qwen3Config.model_type)
            text_model_config = TEXT_MODEL_CONFIGS[model_type](**text_model_config)

        # Convert sub-configs from dict or generic PretrainedConfig to specific types.
        # The generic PretrainedConfig case occurs when transformers' sub_configs mechanism
        # auto-deserializes before __init__ runs (e.g. with trust_remote_code Hub loading).
        def _to_dict(cfg: Any) -> dict[str, Any]:
            """Convert a config to dict, handling both dict and PretrainedConfig."""
            if isinstance(cfg, dict):
                return cfg
            return cfg.to_dict()

        if isinstance(audio_encoder_config, dict) or (
            isinstance(audio_encoder_config, PretrainedConfig)
            and not isinstance(audio_encoder_config, (Qwen3OmniMoeAudioEncoderConfig, VoxtralRealtimeEncoderConfig))
        ):
            d = _to_dict(audio_encoder_config)
            model_type = d.get("model_type", Qwen3OmniMoeAudioEncoderConfig.model_type)
            if model_type == Qwen3OmniMoeAudioEncoderConfig.model_type:
                audio_encoder_config = Qwen3OmniMoeAudioEncoderConfig(**d)
            elif model_type == "voxtral_realtime_encoder":
                audio_encoder_config = VoxtralRealtimeEncoderConfig(**d)
            else:
                raise ValueError(
                    f"Unsupported audio_encoder model_type: {model_type!r}. "
                    "Expected 'qwen3_omni_moe_audio_encoder' or 'voxtral_realtime_encoder'."
                )

        if isinstance(audio_tokenizer_config, dict) or (
            isinstance(audio_tokenizer_config, PretrainedConfig) and not isinstance(audio_tokenizer_config, MimiConfig)
        ):
            audio_tokenizer_config = MimiConfig(**_to_dict(audio_tokenizer_config))

        if isinstance(input_adaptor_config, dict) or (
            isinstance(input_adaptor_config, PretrainedConfig)
            and not isinstance(input_adaptor_config, EmbeddingAdaptorConfig)
        ):
            input_adaptor_config = EmbeddingAdaptorConfig(**_to_dict(input_adaptor_config))

        if isinstance(output_adaptor_config, dict) or (
            isinstance(output_adaptor_config, PretrainedConfig)
            and not isinstance(output_adaptor_config, EmbeddingAdaptorConfig)
        ):
            output_adaptor_config = EmbeddingAdaptorConfig(**_to_dict(output_adaptor_config))

        if isinstance(code_predictor_config, dict) or (
            isinstance(code_predictor_config, PretrainedConfig)
            and not isinstance(code_predictor_config, Qwen3OmniMoeTalkerCodePredictorConfig)
        ):
            code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig(**_to_dict(code_predictor_config))

        if isinstance(speaker_encoder_config, dict) or (
            isinstance(speaker_encoder_config, PretrainedConfig)
            and not isinstance(speaker_encoder_config, SpeakerEncoderConfig)
        ):
            speaker_encoder_config = SpeakerEncoderConfig(**_to_dict(speaker_encoder_config))

        if isinstance(talker_config, dict) or (
            isinstance(talker_config, PretrainedConfig) and type(talker_config) is PretrainedConfig
        ):
            d = _to_dict(talker_config) if talker_config is not None else {}
            talker_model_type = d.get("model_type", Qwen3Config.model_type)
            talker_config = TEXT_MODEL_CONFIGS[talker_model_type](**d)

        assert isinstance(
            audio_encoder_config, (Qwen3OmniMoeAudioEncoderConfig, VoxtralRealtimeEncoderConfig, MimiConfig)
        ), "audio_encoder_config must be Qwen3OmniMoeAudioEncoderConfig, VoxtralRealtimeEncoderConfig, or MimiConfig."
        assert isinstance(audio_tokenizer_config, MimiConfig), "audio_tokenizer_config must be MimiConfig."
        assert isinstance(input_adaptor_config, EmbeddingAdaptorConfig), (
            "input_adaptor_config must be EmbeddingAdaptorConfig."
        )
        assert isinstance(output_adaptor_config, EmbeddingAdaptorConfig), (
            "output_adaptor_config must be EmbeddingAdaptorConfig."
        )
        assert isinstance(code_predictor_config, Qwen3OmniMoeTalkerCodePredictorConfig), (
            "code_predictor_config must be Qwen3OmniMoeTalkerCodePredictorConfig."
        )
        assert isinstance(text_model_config, PretrainedConfig), "text_model_config must be PretrainedConfig."
        assert speaker_encoder_config is None or isinstance(speaker_encoder_config, SpeakerEncoderConfig), (
            "speaker_encoder_config must be None or SpeakerEncoderConfig."
        )

        self.text_model_config = text_model_config
        self.audio_encoder_config = audio_encoder_config
        self.audio_tokenizer_config = audio_tokenizer_config
        self.input_adaptor_config = input_adaptor_config
        self.output_adaptor_config = output_adaptor_config
        self.code_predictor_config = code_predictor_config
        self.speaker_encoder_config = speaker_encoder_config
        self.num_talker_layers = num_talker_layers
        self.supports_audio_input = supports_audio_input
        self.supports_audio_output = supports_audio_output
        self.aut_is_causal = aut_is_causal
        self.proj_code_bias = proj_code_bias
        self.accept_hidden_layer = accept_hidden_layer
        self.talker_config = talker_config
        self.thinker_to_talker_pre_norm = thinker_to_talker_pre_norm
        self.sequence_mode = sequence_mode
        self.use_sil_token = use_sil_token
        self.no_audio_in_sil = no_audio_in_sil
        self.text_lookahead = int(text_lookahead)
        self.use_duplex_end_pad = use_duplex_end_pad
        self.speaker_embedding_to_code_predictor = speaker_embedding_to_code_predictor
        self.duplex_pad_token_id = duplex_pad_token_id
        self.duplex_end_pad_token_id = duplex_end_pad_token_id
        self.duplex_sil_token_id = duplex_sil_token_id
        self.duplex_bc_token_id = duplex_bc_token_id
        self.use_backchannel_token = use_backchannel_token
        self.bc_loss_weight = bc_loss_weight
        self.speaker_token_id = speaker_token_id
        self.audio_input_token_id = audio_input_token_id
        self.audio_output_token_id = audio_output_token_id
        self.audio_start_token_id = audio_start_token_id
        self.im_start_token_id = im_start_token_id
        self.text_loss_weight = text_loss_weight
        self.sil_loss_weight = sil_loss_weight
        self.epad_loss_weight = epad_loss_weight
        self.semantic_loss_weight = semantic_loss_weight
        self.acoustic_loss_weights = acoustic_loss_weights
        self.audio_lm_head_enabled = audio_lm_head_enabled
        self.delays = delays

        if supports_audio_output and audio_lm_head_enabled:
            assert talker_config is not None, "RaonConfig: `talker_config` is required when audio output is enabled."
            assert num_talker_layers > 0, "RaonConfig: `num_talker_layers` must be positive when audio output is enabled."

    def _get_non_default_generation_parameters(self) -> dict[str, Any]:
        return {}

    def to_diff_dict(self) -> dict[str, Any]:
        """Return config as a dict suitable for diffing."""
        return self.to_dict()


@dataclass
class RaonModelOutput(ModelOutput):
    """Output container for the training forward pass."""

    loss: torch.Tensor | None = None
    text_loss: torch.Tensor | None = None
    audio_loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    text_last_hidden_state: torch.Tensor | None = None
    talker_last_hidden_state: torch.Tensor | None = None
    text_logits: torch.Tensor | None = None
    audio_logits: torch.Tensor | None = None
    router_logits: tuple[torch.Tensor, ...] | None = None
    past_key_values: Cache | None = None


class RaonModel(RaonLossMixin, PreTrainedModel, RaonInferenceModel):
    """Core raon model combining text language model with audio codec."""

    _tied_weights_keys: list[str] = []  # type: ignore
    config_class: type[RaonConfig] = RaonConfig  # type: ignore
    config: RaonConfig

    def __init__(self, config: RaonConfig) -> None:
        super().__init__(config)
        assert config.text_model_config is not None, "Config text_model_config is required."
        assert config.audio_encoder_config is not None, "Config audio_encoder_config is required."
        assert config.audio_tokenizer_config is not None, "Config audio_tokenizer_config is required."
        assert config.input_adaptor_config is not None, "Config input_adaptor_config is required."
        assert config.output_adaptor_config is not None, "Config output_adaptor_config is required."
        assert config.code_predictor_config is not None, "Config code_predictor_config is required."
        assert config.text_model_config.vocab_size is not None, "text_model_config.vocab_size is required."
        assert config.audio_tokenizer_config.codebook_size is not None, "audio_tokenizer_config.codebook_size is required."
        assert config.code_predictor_config.num_code_groups is not None, "code_predictor_config.num_code_groups is required."
        assert config.audio_tokenizer_config.sampling_rate is not None, "audio_tokenizer_config.sampling_rate is required."
        assert config.text_model_config.hidden_size is not None, "text_model_config.hidden_size is required."
        assert config.code_predictor_config.hidden_size is not None, "code_predictor_config.hidden_size is required."
        if getattr(config, "supports_audio_output", True) and getattr(config, "audio_lm_head_enabled", True):
            assert config.talker_config is not None, "talker_config is required when audio output is enabled."
            assert config.num_talker_layers > 0, "num_talker_layers must be positive when audio output is enabled."

        self.config = config
        _dtype = getattr(config, "torch_dtype", None) or torch.float32
        if isinstance(_dtype, str):
            _dtype = getattr(torch, _dtype, torch.float32)
        self.hidden_size = int(config.text_model_config.hidden_size)
        self.vocab_size = int(config.text_model_config.vocab_size)
        self.codebook_size = config.audio_tokenizer_config.codebook_size
        self.audio_lm_head_vocab_size = self.codebook_size + 1
        self.num_talker_layers = config.num_talker_layers
        self.supports_audio_input = getattr(config, "supports_audio_input", True)
        self.supports_audio_output = getattr(config, "supports_audio_output", True)
        self.num_code_groups = config.code_predictor_config.num_code_groups
        self.sampling_rate = config.audio_tokenizer_config.sampling_rate
        assert (frame_rate := config.audio_tokenizer_config._frame_rate) is not None, (  # type: ignore
            "audio_tokenizer_config._frame_rate is required."
        )
        self.frame_rate = frame_rate
        self.output_losses_only = False
        self._debug_tokenizer: Any | None = None
        self._debug_tokenizer_disabled = False

        # Create thinker text_model: num_hidden_layers IS the thinker count (talker is separate).
        total_layers = int(config.text_model_config.num_hidden_layers)
        num_thinker_layers = total_layers
        thinker_text_config = deepcopy(config.text_model_config)
        thinker_text_config.num_hidden_layers = num_thinker_layers
        if hasattr(thinker_text_config, "layer_types") and thinker_text_config.layer_types:
            thinker_text_config.layer_types = thinker_text_config.layer_types[:num_thinker_layers]
        self.text_model = TEXT_MODELS[thinker_text_config.model_type]._from_config(
            thinker_text_config,
            dtype=_dtype,
        )
        if self.supports_audio_input:
            # Use model_type string instead of isinstance to avoid class identity issues
            # with trust_remote_code dynamic module loading.
            ae_model_type = getattr(config.audio_encoder_config, "model_type", "")
            if ae_model_type == "mimi":
                self.audio_encoder: CausalAudioEncoder | AuTWrapper | VoxtralWrapper | None = (
                    CausalAudioEncoder._from_config(
                        config.audio_encoder_config,
                        dtype=_dtype,
                    )
                )
            elif ae_model_type == "voxtral_realtime_encoder":
                self.audio_encoder: CausalAudioEncoder | AuTWrapper | VoxtralWrapper | None = VoxtralWrapper.from_config(
                    config=config.audio_encoder_config,
                    dtype=_dtype,
                )
            else:
                assert ae_model_type in ("qwen3_omni_moe_audio_encoder", ""), (
                    f"RAON checkpoints require audio_encoder model_type 'qwen3_omni_moe_audio_encoder' or "
                    f"'voxtral_realtime_encoder', got {ae_model_type!r}."
                )
                self.audio_encoder: CausalAudioEncoder | AuTWrapper | VoxtralWrapper | None = AuTWrapper.from_config(
                    config=config.audio_encoder_config,
                    dtype=_dtype,
                )
        else:
            self.audio_encoder = None

        self.aut_is_causal = getattr(config, "aut_is_causal", False)

        if self.supports_audio_output:
            self.audio_tokenizer: StreamingMimiModel | None = StreamingMimiModel._from_config(
                config.audio_tokenizer_config,
                dtype=_dtype,
            )
        else:
            self.audio_tokenizer = None
        if self.supports_audio_input:
            self.input_adaptor: EmbeddingAdaptor | None = EmbeddingAdaptor.from_config(
                config.input_adaptor_config, dtype=_dtype
            )
        else:
            self.input_adaptor = None
        if self.supports_audio_output:
            self.output_adaptor: EmbeddingAdaptor | None = EmbeddingAdaptor.from_config(
                config.output_adaptor_config, dtype=_dtype
            )
        else:
            self.output_adaptor = None
        # Create separate talker model and thinker-to-talker projection (audio output only).
        rms_norm_eps = getattr(config.text_model_config, "rms_norm_eps", 1e-6)
        if self.supports_audio_output and config.talker_config is not None:
            resolved_talker_config = config.talker_config
            self.talker: PreTrainedModel | None = TEXT_MODELS[resolved_talker_config.model_type]._from_config(
                resolved_talker_config,
                dtype=_dtype,
            )
            # Talker only receives inputs_embeds (from thinker_to_talker_proj), never input_ids.
            self.talker.embed_tokens = None  # type: ignore
            talker_hidden_size = int(resolved_talker_config.hidden_size)
            projection_mode = getattr(config, "thinker_to_talker_projection_mode", "linear")
            projection_intermediate_size = getattr(config, "thinker_to_talker_intermediate_size", None)
            if projection_mode == "mlp" and projection_intermediate_size is None:
                projection_intermediate_size = int(resolved_talker_config.intermediate_size)
            self.thinker_to_talker_proj: ThinkerToTalkerProjection | None = ThinkerToTalkerProjection(
                thinker_hidden_size=self.hidden_size,
                talker_hidden_size=talker_hidden_size,
                intermediate_size=projection_intermediate_size,
                mode=projection_mode,
                use_norm=getattr(config, "thinker_to_talker_pre_norm", False),
                rms_norm_eps=rms_norm_eps,
            )
        else:
            self.talker = None
            self.thinker_to_talker_proj = None
            talker_hidden_size = self.hidden_size

        # Resolve accept_hidden_layer: -1 means last thinker layer.
        accept_hidden_layer = getattr(config, "accept_hidden_layer", -1)
        if accept_hidden_layer < 0:
            accept_hidden_layer = num_thinker_layers + accept_hidden_layer
        self.accept_hidden_layer = accept_hidden_layer
        self.thinker_capture_layer_index = (
            accept_hidden_layer  # Index of the thinker layer whose output feeds the talker and audio head.
        )
        self.lm_head = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.vocab_size,
            bias=False,
            dtype=_dtype,
        )
        if self.supports_audio_output:
            self.audio_lm_head: nn.Linear | None = nn.Linear(
                in_features=talker_hidden_size,
                out_features=self.audio_lm_head_vocab_size,
                bias=False,
                dtype=_dtype,
            )
            if not getattr(config, "audio_lm_head_enabled", True):
                self.audio_lm_head = None
            self.proj_code: nn.Linear | None = nn.Linear(
                in_features=talker_hidden_size,
                out_features=config.code_predictor_config.hidden_size,
                bias=config.proj_code_bias,
                dtype=_dtype,
            )
            self.code_predictor: RaonCodePredictorModel | None = RaonCodePredictorModel._from_config(
                config.code_predictor_config,
                dtype=_dtype,
            )
        else:
            self.audio_lm_head = None
            self.proj_code = None
            self.code_predictor = None

        self.accepted_thinker_hidden_states: torch.Tensor | None = None
        self.register_thinker_capture_hook()

        self.code_predictor_grad_scale = _read_loss_param(
            env_key="RAON_CODE_PREDICTOR_GRAD_SCALE",
            config=config,
            attr_name="code_predictor_grad_scale",
            default=0.1,
        )
        self.text_loss_weight = _read_loss_param(
            env_key="RAON_TEXT_LOSS_WEIGHT",
            config=config,
            attr_name="text_loss_weight",
            default=1.0,
        )
        self.audio_output_pad_text_loss_weight = _read_loss_param(
            env_key="RAON_AUDIO_OUTPUT_PAD_TEXT_LOSS_WEIGHT",
            config=config,
            attr_name="audio_output_pad_text_loss_weight",
            default=0.0,
        )
        self.epad_loss_weight = _read_loss_param(
            env_key="RAON_EPAD_LOSS_WEIGHT",
            config=config,
            attr_name="epad_loss_weight",
            default=1.0,
        )
        self.audio_end_text_loss_weight = _read_loss_param(
            env_key="RAON_AUDIO_END_TEXT_LOSS_WEIGHT",
            config=config,
            attr_name="audio_end_text_loss_weight",
            default=0.0,
        )
        self.semantic_loss_weight = _read_loss_param(
            env_key="RAON_SEMANTIC_LOSS_WEIGHT",
            config=config,
            attr_name="semantic_loss_weight",
            default=1.0,
        )
        self.speaker_dropout = _read_loss_param(
            env_key="RAON_SPEAKER_DROPOUT",
            config=config,
            attr_name="speaker_dropout",
            default=0.2,
        )
        acoustic_loss_weights = _read_acoustic_loss_weights(config=config, num_code_groups=self.num_code_groups)
        self.audio_loss_weight = torch.tensor([self.semantic_loss_weight] + acoustic_loss_weights, dtype=_dtype)

        # Speaker encoder (optional, for speaker-conditioned TTS)
        if self.supports_audio_output and config.speaker_encoder_config is not None:
            self.speaker_encoder: PretrainedSpeakerEncoder | None = PretrainedSpeakerEncoder(
                config.speaker_encoder_config,
                dtype=_dtype,
            )
            self.is_pretrained_speaker_encoder = True
            self.speaker_token_id: int | None = SPEAKER_EMBEDDING_PLACEHOLDER.id
        else:
            self.speaker_encoder = None
            self.is_pretrained_speaker_encoder = False
            self.speaker_token_id = getattr(config, "speaker_token_id", SPEAKER_EMBEDDING_PLACEHOLDER.id)

        # Defensive cleanup: when audio output is disabled, ensure all output-side
        # audio modules are hard-disabled even if something assigned them earlier.
        if not self.supports_audio_output:
            self._disable_audio_output_modules()
        if not self.supports_audio_input:
            self._disable_audio_input_modules()

        if self.config.input_adaptor_config.output_time_scale != 1:
            raise NotImplementedError("Only `output_time_scale == 1` is supported.")

        RaonInferenceModel.__init__(self)

        # Duplex runtime attributes
        self.use_duplex_end_pad = getattr(config, "use_duplex_end_pad", False)
        self.use_sil_token = getattr(config, "use_sil_token", False)
        self.no_audio_in_sil = getattr(config, "no_audio_in_sil", False)
        self.sequence_mode = getattr(config, "sequence_mode", None)
        self.duplex_pad_token_id = getattr(config, "duplex_pad_token_id", AUDIO_OUTPUT_PAD.id)
        self.duplex_end_pad_token_id = getattr(config, "duplex_end_pad_token_id", AUDIO_OUTPUT_END_PAD.id)
        self.duplex_sil_token_id = getattr(config, "duplex_sil_token_id", -1)
        self.use_backchannel_token = getattr(config, "use_backchannel_token", False)
        self.duplex_bc_token_id = getattr(config, "duplex_bc_token_id", AUDIO_OUTPUT_BC.id)
        self.bc_loss_weight = getattr(config, "bc_loss_weight", 1.0)
        self.audio_start_token_id = getattr(config, "audio_start_token_id", AUDIO_START.id)
        self.im_start_token_id = getattr(config, "im_start_token_id", IM_START.id)
        self.audio_input_token_id = getattr(config, "audio_input_token_id", AUDIO_INPUT_PLACEHOLDER.id)
        self.audio_output_token_id = getattr(config, "audio_output_token_id", AUDIO_OUTPUT_PLACEHOLDER.id)
        self.delays = getattr(config, "delays", None) or [0] * self.num_code_groups
        self.max_delay = max(self.delays)
        self.text_vocab_size = self.vocab_size - self.codebook_size
        # Loss weights (overridable at training time via duplex_train args)
        self.text_loss_weight = getattr(config, "text_loss_weight", 1.0)
        self.sil_loss_weight = getattr(config, "sil_loss_weight", 1.0)
        self.epad_loss_weight = getattr(config, "epad_loss_weight", 0.0)
        self.semantic_loss_weight = getattr(config, "semantic_loss_weight", 1.0)
        self.acoustic_loss_weights = getattr(config, "acoustic_loss_weights", None)

    def _disable_audio_output_modules(self) -> None:
        """Force-disable all audio-output and speaker-output modules."""
        self.audio_tokenizer = None
        self.output_adaptor = None
        self.audio_lm_head = None
        self.proj_code = None
        self.code_predictor = None
        self.speaker_encoder = None
        self.is_pretrained_speaker_encoder = False
        self.speaker_token_id = None

    def _disable_audio_input_modules(self) -> None:
        """Force-disable all audio-input modules."""
        self.audio_encoder = None
        self.input_adaptor = None

    def register_thinker_capture_hook(self) -> None:
        """Register a forward hook on the accept_hidden_layer to capture unnormalized hidden states."""

        def hook(module: nn.Module, input: Any, output: Any) -> None:
            self.accepted_thinker_hidden_states = output[0] if isinstance(output, tuple) else output

        cast(list[nn.Module], self.text_model.layers)[self.accept_hidden_layer].register_forward_hook(hook)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the text model input embedding layer."""
        assert isinstance(self.text_model.embed_tokens, nn.Embedding), "text_model.embed_tokens must be nn.Embedding."
        return self.text_model.embed_tokens

    def _validate_audio_output_inputs(
        self,
        audio_output: torch.Tensor | None,
        audio_output_codes: torch.Tensor | None,
        audio_output_codes_mask: torch.Tensor | None,
    ) -> None:
        """Validate that audio-output inputs are only used when audio output is supported."""
        if self.supports_audio_output:
            return
        if audio_output is not None or audio_output_codes is not None or audio_output_codes_mask is not None:
            raise ValueError(
                "Audio output is disabled (`supports_audio_output=False`), but audio-output inputs were provided."
            )

    def _validate_audio_input_inputs(
        self,
        input_ids: torch.Tensor | None,
        audio_input: torch.Tensor | None,
        audio_input_embeds: torch.Tensor | None,
        audio_input_embeds_mask: torch.Tensor | None,
    ) -> None:
        """Validate that audio-input inputs are only used when audio input is supported."""
        if self.supports_audio_input:
            return
        _ = input_ids, audio_input
        if audio_input_embeds is not None or audio_input_embeds_mask is not None:
            raise ValueError("Audio input is disabled (`supports_audio_input=False`), but audio-input inputs were provided.")

    def get_model(self) -> Self:
        """Return self as the model instance."""
        return self

    def tie_weights(
        self,
        missing_keys: set[str] | None = None,
        recompute_mapping: bool = False,
    ) -> None: ...

    @property
    def all_tied_weights_keys(self) -> dict[str, str]:
        return {}

    @staticmethod
    def _normalize_audio_dims(audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio tensor to 3D [batch_size, num_channels, num_samples].

        Accepts 1D [num_samples], 2D [batch_size, num_samples], or 3D
        [batch_size, num_channels, num_samples] inputs and returns the 3D form.
        """
        if audio.ndim == 1:
            audio = audio[None, None]
        elif audio.ndim == 2:
            audio = audio[:, None]
        assert audio.ndim == 3, "Audio tensor must have 3 dimensions [batch_size, num_channels, num_samples]."
        return audio

    def get_audio_input_embeds(
        self,
        audio: torch.Tensor | None = None,
        audio_lengths: torch.Tensor | None = None,
        sampling_rate: int | None = None,
        num_code_groups: int = 8,
        encoder_past_key_values: Cache | None = None,
        conv_padding_cache: MimiConv1dPaddingCache | None = None,
        use_streaming: bool | None = None,
    ) -> AudioEncoderOutput:
        """Encode raw audio to input embeddings via audio encoder and input adaptor.

        Args:
            audio: Raw waveform. Shape: [batch_size, num_channels, num_samples] or
                [batch_size, num_samples] or [num_samples]. Dtype: float. None to return empty output.
            audio_lengths: Valid length per sample. Shape: [batch_size]. Dtype: long.
            sampling_rate: Sample rate of input audio; resampled if different from encoder.
            num_code_groups: Number of code groups (unused, for API compatibility).
            encoder_past_key_values: Cached encoder KV for streaming.
            conv_padding_cache: Cached conv padding for streaming encoder.
            use_streaming: Whether to use streaming mode.

        Returns:
            AudioEncoderOutput with audio_embeds (Shape: [batch_size, num_frames, feature_dim].
            Dtype: float.) and audio_embeds_mask (Shape: [batch_size, num_frames]. Dtype: bool.).
        """
        if audio is None:
            return AudioEncoderOutput()
        if self.audio_encoder is None:
            raise RuntimeError("audio_encoder is unavailable when supports_audio_input is False.")
        assert self.input_adaptor is not None, "input_adaptor is unavailable when supports_audio_input is False."

        audio = cast_to_module_dtype(audio, self.audio_encoder)
        audio = RaonModel._normalize_audio_dims(audio)

        if sampling_rate is not None and sampling_rate != self.audio_encoder.config.sampling_rate:
            assert self.audio_encoder.config.sampling_rate is not None, (
                "audio_encoder.config.sampling_rate is required for resampling."
            )
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=sampling_rate,
                new_freq=self.audio_encoder.config.sampling_rate,
            )

        encoder_cache: tuple[Cache, MimiConv1dPaddingCache] | None = None
        if isinstance(self.audio_encoder, CausalAudioEncoder):
            encoder_outputs = self.audio_encoder(
                audio,
                encoder_past_key_values=encoder_past_key_values,
                padding_cache=conv_padding_cache,
                use_streaming=use_streaming,
            )
            assert isinstance(encoder_outputs, CausalAudioEncoderOutput), "encoder_outputs must be CausalAudioEncoderOutput."
            if encoder_outputs.encoder_past_key_values is not None and encoder_outputs.padding_cache is not None:
                assert isinstance(encoder_outputs.encoder_past_key_values, Cache), "encoder_past_key_values must be Cache."
                assert isinstance(encoder_outputs.padding_cache, MimiConv1dPaddingCache), (
                    "padding_cache must be MimiConv1dPaddingCache."
                )
                encoder_cache = (
                    encoder_outputs.encoder_past_key_values,
                    encoder_outputs.padding_cache,
                )
        else:
            encoder_kwargs: dict[str, Any] = {}
            if isinstance(self.audio_encoder, (AuTWrapper, VoxtralWrapper)):
                encoder_kwargs["causal"] = self.aut_is_causal
                encoder_kwargs["audio_lengths"] = audio_lengths
            encoder_outputs = self.audio_encoder(audio, **encoder_kwargs)

        assert (audio_embeds := encoder_outputs.embeds) is not None, "Encoder outputs must contain embeds."

        if audio_lengths is not None:
            indices = torch.arange(audio.shape[-1], device=audio.device)
            audio_embeds_mask = (indices[None] < audio_lengths[:, None]).long()

            assert (encoder_sampling_rate := self.config.audio_tokenizer_config.sampling_rate) is not None, (
                "audio_tokenizer_config.sampling_rate is required."
            )
            assert (frame_rate := self.config.audio_tokenizer_config._frame_rate) is not None, (  # type: ignore
                "audio_tokenizer_config._frame_rate is required."
            )
            assert (samples_per_frame := int(encoder_sampling_rate / frame_rate)) == encoder_sampling_rate / frame_rate, (
                "samples_per_frame must divide evenly."
            )
            padded_audio_mask = F.pad(
                audio_embeds_mask,
                (0, audio_embeds.shape[1] * samples_per_frame - audio_embeds_mask.shape[1]),
            )
            audio_embeds_mask = padded_audio_mask.view(-1, audio_embeds.shape[1], samples_per_frame).any(dim=-1)
        else:
            audio_embeds_mask = torch.ones(
                audio_embeds.shape[:2],
                dtype=torch.bool,
                device=audio_embeds.device,
            )

        adaptor_outputs = self.input_adaptor(audio_embeds, mask=audio_embeds_mask)
        assert isinstance(adaptor_outputs, EmbeddingAdaptorOutput), "adaptor_outputs must be EmbeddingAdaptorOutput."
        assert (audio_embeds := adaptor_outputs.outputs_embeds) is not None, "adaptor outputs_embeds is required."
        assert (audio_embeds_mask := adaptor_outputs.mask) is not None, "adaptor mask is required."  # type: ignore

        return AudioEncoderOutput(
            audio_embeds=audio_embeds,
            audio_embeds_mask=audio_embeds_mask,
            encoder_cache=encoder_cache,
        )

    def tokenize_audio(
        self,
        audio: torch.Tensor | None = None,
        audio_lengths: torch.Tensor | None = None,
        sampling_rate: int | None = None,
        num_code_groups: int = 8,
        return_mimi_features: bool = False,
        encoder_past_key_values: Cache | None = None,
        conv_padding_cache: MimiConv1dPaddingCache | None = None,
        use_streaming: bool | None = None,
    ) -> AudioTokenizerOutput:
        if audio is None:
            return AudioTokenizerOutput()
        if self.audio_tokenizer is None:
            raise RuntimeError("audio_tokenizer is unavailable when supports_audio_output is False.")
        target_sampling_rate = (
            self.audio_encoder.config.sampling_rate
            if self.audio_encoder is not None
            else self.audio_tokenizer.config.sampling_rate
        )
        # Cast to audio_tokenizer (Mimi) dtype — the encoder dtype may differ (e.g. VoxtralWrapper).
        audio = cast_to_module_dtype(audio, self.audio_tokenizer)
        audio = RaonModel._normalize_audio_dims(audio)

        if sampling_rate is not None and sampling_rate != target_sampling_rate:
            assert target_sampling_rate is not None, "sampling_rate is required for resampling."
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=sampling_rate,
                new_freq=target_sampling_rate,
            )

        audio_mask = None
        if audio_lengths is not None:
            indices = torch.arange(audio.shape[-1], device=audio.device)
            audio_mask = (indices[None] < audio_lengths[:, None]).long()

        outputs = self.audio_tokenizer.encode(
            audio,
            padding_mask=audio_mask,
            num_quantizers=num_code_groups,
            encoder_past_key_values=encoder_past_key_values,
            padding_cache=conv_padding_cache,
            use_streaming=use_streaming,
            return_dict=True,
        )
        assert isinstance(outputs, MimiEncoderOutput), "tokenizer encode output must be MimiEncoderOutput."

        encoder_cache: tuple[Cache, MimiConv1dPaddingCache] | None = None
        if outputs.encoder_past_key_values is not None and outputs.padding_cache is not None:
            assert isinstance(outputs.encoder_past_key_values, Cache), "encoder_past_key_values must be Cache."
            assert isinstance(outputs.padding_cache, MimiConv1dPaddingCache), "padding_cache must be MimiConv1dPaddingCache."
            encoder_cache = (outputs.encoder_past_key_values, outputs.padding_cache)

        assert outputs.audio_codes is not None, "tokenizer encode output must contain audio_codes."
        audio_codes = outputs.audio_codes.view(outputs.audio_codes.shape[-3:]).transpose(1, 2)
        if audio_mask is not None:
            assert (encoder_sampling_rate := self.config.audio_tokenizer_config.sampling_rate) is not None, (
                "audio_tokenizer_config.sampling_rate is required."
            )
            assert (frame_rate := self.config.audio_tokenizer_config._frame_rate) is not None, (  # type: ignore
                "audio_tokenizer_config._frame_rate is required."
            )
            assert (samples_per_frame := int(encoder_sampling_rate / frame_rate)) == encoder_sampling_rate / frame_rate, (
                "samples_per_frame must divide evenly."
            )
            padded_audio_mask = F.pad(
                audio_mask,
                (0, audio_codes.shape[1] * samples_per_frame - audio_mask.shape[1]),
            )
            audio_codes_mask = padded_audio_mask.view(-1, audio_codes.shape[1], samples_per_frame).any(dim=-1)
        else:
            audio_codes_mask = torch.ones(
                audio_codes.shape[:2],
                dtype=torch.bool,
                device=audio_codes.device,
            )

        # Optionally return mimi_features for speaker embedding
        mimi_features = None
        if return_mimi_features:
            # Get latent features by decoding the audio codes (quantizer decode)
            # Shape: [batch_size, 512, num_frames] -> transpose to [batch_size, num_frames, 512]
            mimi_features = self.audio_tokenizer.quantizer.decode(audio_codes.transpose(1, 2)).transpose(1, 2)

        return AudioTokenizerOutput(
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            mimi_features=mimi_features,
            encoder_cache=encoder_cache,
        )

    def tokenize_audio_segments(
        self,
        audio: torch.Tensor,
        segments: list[tuple[int, int]],
        num_code_groups: int = 8,
        return_mimi_features: bool = False,
    ) -> AudioTokenizerOutput:
        """Tokenize speech segments independently via batched Mimi encoding.

        Each segment is encoded independently (no cross-segment conv context).
        Used in no_audio_in_sil mode where SIL frames have no [A] token.

        Args:
            audio: Raw waveform. Shape: [1, num_samples] or [1, 1, num_samples]. Dtype: float.
            segments: List of (start_sample, end_sample) tuples for utterance regions.
            num_code_groups: Number of codec groups for Mimi quantizer.
            return_mimi_features: Whether to return pre-quantization features.

        Returns:
            AudioTokenizerOutput with codes for speech frames only (concatenated).
        """
        if not segments:
            return AudioTokenizerOutput()

        if audio.ndim == 3:
            audio = audio.squeeze(1)

        assert self.config.audio_tokenizer_config.sampling_rate is not None
        assert self.config.audio_tokenizer_config._frame_rate is not None  # type: ignore
        sr = self.config.audio_tokenizer_config.sampling_rate
        fr = self.config.audio_tokenizer_config._frame_rate  # type: ignore

        seg_audios = [audio[0, s:e] for s, e in segments]
        seg_lengths = torch.tensor([s.shape[0] for s in seg_audios], device=audio.device)

        max_len = int(seg_lengths.max().item())
        batched = torch.stack([F.pad(s, (0, max_len - s.shape[0])) for s in seg_audios]).to(audio.dtype)

        batched_out = self.tokenize_audio(
            audio=batched,
            audio_lengths=seg_lengths,
            num_code_groups=num_code_groups,
            return_mimi_features=return_mimi_features,
        )
        if batched_out.audio_codes is None:
            return AudioTokenizerOutput()

        all_codes = []
        all_masks = []
        all_features = []
        for i, length in enumerate(seg_lengths):
            n_frames = math.ceil(length.item() * fr / sr)
            n_frames = min(n_frames, batched_out.audio_codes.shape[1])
            all_codes.append(batched_out.audio_codes[i : i + 1, :n_frames])
            if batched_out.audio_codes_mask is not None:
                all_masks.append(batched_out.audio_codes_mask[i : i + 1, :n_frames])
            if batched_out.mimi_features is not None:
                all_features.append(batched_out.mimi_features[i : i + 1, :n_frames])

        audio_codes = torch.cat(all_codes, dim=1)
        audio_codes_mask = torch.cat(all_masks, dim=1) if all_masks else None
        mimi_features = torch.cat(all_features, dim=1) if all_features else None

        if audio_codes_mask is None:
            audio_codes_mask = torch.ones(audio_codes.shape[:2], dtype=torch.bool, device=audio_codes.device)

        return AudioTokenizerOutput(
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            mimi_features=mimi_features,
        )

    def _get_audio_output_embeds(
        self,
        audio_codes: torch.Tensor,
        audio_codes_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self.audio_tokenizer is not None, "audio_tokenizer is unavailable when supports_audio_output is False."
        assert self.output_adaptor is not None, "output_adaptor is unavailable when supports_audio_output is False."
        assert audio_codes.ndim == 3 and audio_codes_mask.ndim == 2, (
            "audio_codes must have 3 dims and audio_codes_mask must have 2 dims."
        )
        latent_features = self.audio_tokenizer.quantizer.decode(audio_codes.transpose(1, 2)).transpose(1, 2)
        adaptor_outputs = self.output_adaptor(latent_features, mask=audio_codes_mask)
        return adaptor_outputs.outputs_embeds, adaptor_outputs.mask

    def _insert_audio_embeds(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        audio_embeds: torch.Tensor,
        audio_embeds_mask: torch.Tensor | None,
        audio_token_id: int,
    ) -> torch.Tensor:
        audio_mask = (input_ids == audio_token_id)[..., None].expand_as(inputs_embeds)
        audio_embeds = cast_float_inputs(audio_embeds, inputs_embeds.dtype)
        if audio_embeds_mask is not None:
            audio_embeds = audio_embeds[audio_embeds_mask]
        else:
            audio_embeds = audio_embeds.view(-1, audio_embeds.shape[-1])

        assert audio_mask.sum() == audio_embeds.numel(), (
            f"Number of masked positions must match audio_embeds element count. "
            f"audio_mask.sum()={audio_mask.sum()}, audio_embeds.numel()={audio_embeds.numel()}, "
            f"audio_token_id={audio_token_id}, input_ids_shape={input_ids.shape}."
        )
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
        return inputs_embeds

    def update_inputs_embeds(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        audio_output_codes: torch.Tensor | None = None,
        audio_output_codes_mask: torch.Tensor | None = None,
        speaker_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Insert audio input, audio output, and speaker embeddings into inputs_embeds at placeholder positions.

        Args:
            inputs_embeds: Base embeddings from text tokens. Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            input_ids: Token IDs for locating placeholders. Shape: [batch_size, seq_length]. Dtype: long.
            audio_input_embeds: Encoded audio input embeddings. Shape: [batch_size, num_frames, hidden_size].
                Dtype: float.
            audio_input_embeds_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            audio_output_codes: Discrete audio output codes. Shape: [batch_size, num_frames, num_code_groups].
                Dtype: long.
            audio_output_codes_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            speaker_embeds: Speaker conditioning embeddings. Shape: [batch_size, 1, hidden_size]. Dtype: float.

        Returns:
            Updated inputs_embeds with all placeholders filled.
            Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
        """
        if audio_output_codes is not None:
            assert input_ids is not None, "`input_ids` required when `audio_output_codes` is provided."
            assert audio_output_codes_mask is not None, (
                "`audio_output_codes_mask` required when `audio_output_codes` is provided."
            )

        if audio_input_embeds is not None:
            assert input_ids is not None, "`input_ids` required when `audio_input_embeds` is provided."
            assert audio_input_embeds_mask is not None, (
                "`audio_input_embeds_mask` required when `audio_input_embeds` is provided."
            )

        if (
            input_ids is not None
            and audio_input_embeds is not None
            and audio_input_embeds_mask is not None
            and audio_input_embeds_mask.any()
        ):
            inputs_embeds = self._insert_audio_embeds(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                audio_embeds=audio_input_embeds,
                audio_embeds_mask=audio_input_embeds_mask,
                audio_token_id=AUDIO_INPUT_PLACEHOLDER.id,
            )

        if (
            input_ids is not None
            and audio_output_codes is not None
            and audio_output_codes_mask is not None
            and audio_output_codes_mask.any()
        ):
            audio_output_embeds, audio_output_embeds_mask = self._get_audio_output_embeds(
                audio_codes=audio_output_codes,
                audio_codes_mask=audio_output_codes_mask,
            )
            inputs_embeds = self._insert_audio_embeds(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                audio_embeds=audio_output_embeds,
                audio_embeds_mask=audio_output_embeds_mask,
                audio_token_id=AUDIO_OUTPUT_PLACEHOLDER.id,
            )

        # Insert speaker embedding at <|speaker_embedding_placeholder|> position
        if input_ids is not None and speaker_embeds is not None and self.speaker_token_id is not None:
            speaker_mask = input_ids == self.speaker_token_id
            if speaker_mask.any():
                inputs_embeds = self._insert_audio_embeds(
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                    audio_embeds=speaker_embeds,
                    audio_embeds_mask=None,  # Single token per sample, no mask needed
                    audio_token_id=self.speaker_token_id,
                )

        return inputs_embeds

    def shift_labels(self, labels: torch.Tensor, pad_length: int = 1) -> torch.Tensor:
        """Shift labels left by one position for causal LM and optionally pad at the end.

        Args:
            labels: Ground-truth token IDs. Shape: [batch_size, seq_length]. Dtype: long.
            pad_length: Number of LOSS_IGNORE_INDEX values to pad at the end (default 1 for right-aligned targets).

        Returns:
            Shifted labels. Shape: [batch_size, seq_length - 1 + pad_length] when pad_length > 0,
            else [batch_size, seq_length - 1]. Dtype: long.
        """
        if pad_length == 0:
            return labels[:, 1:]
        elif pad_length > 0:
            return F.pad(labels[:, 1:], (0, pad_length), value=LOSS_IGNORE_INDEX)
        else:
            raise ValueError("`pad_length` must be nonnegative.")

    def get_text_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert labels by replacing AUDIO_OUTPUT_PLACEHOLDER with AUDIO_OUTPUT_PAD for text loss.

        Args:
            labels: Ground-truth token IDs including audio placeholders. Shape: [batch_size, seq_length]. Dtype: long.

        Returns:
            Labels suitable for text loss (placeholders replaced with pad). Shape: [batch_size, seq_length]. Dtype: long.
        """
        text_labels = labels.clone()
        text_labels[text_labels == AUDIO_OUTPUT_PLACEHOLDER.id] = AUDIO_OUTPUT_PAD.id
        return text_labels

    def get_proj_code(self) -> nn.Linear:
        """Return the audio code projection layer."""
        assert self.proj_code is not None, "proj_code is unavailable when supports_audio_output is False."
        return self.proj_code

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        audio_input: torch.Tensor | None = None,
        audio_output: torch.Tensor | None = None,
        audio_input_lengths: torch.Tensor | None = None,
        audio_output_lengths: torch.Tensor | None = None,
        speaker_encoder_audio: torch.Tensor | None = None,
        speaker_encoder_audio_lengths: torch.Tensor | None = None,
        audio_output_codes: torch.Tensor | None = None,
        audio_output_codes_mask: torch.Tensor | None = None,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: StaticCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        use_speaker_embedding: bool = False,
        speaker_embeds: torch.Tensor | None = None,
        audio_output_segments: list[tuple[int, int]] | None = None,
        debug_mode: bool = True,
        debug_step: int | None = None,
        **kwargs: Any,
    ) -> RaonModelOutput:
        """Run training forward pass: embed inputs, run text model, compute text and audio loss.

        Args:
            input_ids: Token IDs. Shape: [batch_size, seq_length]. Dtype: long.
            attention_mask: Valid position mask. Shape: [batch_size, seq_length]. Dtype: long.
            audio_input: Raw input audio. Shape: [batch_size, num_channels, num_samples] or [batch_size, num_samples].
                Dtype: float.
            audio_output: Raw output audio for tokenization. Same shapes as audio_input.
            audio_input_lengths: Valid sample lengths for audio_input. Shape: [batch_size]. Dtype: long.
            audio_output_lengths: Valid sample lengths for audio_output. Shape: [batch_size]. Dtype: long.
            speaker_encoder_audio: Unchunked audio for pretrained speaker encoder.
                Shape: [num_speakers, num_samples]. Dtype: float.
            speaker_encoder_audio_lengths: Valid sample lengths for speaker_encoder_audio.
                Shape: [num_speakers]. Dtype: long.
            audio_output_codes: Pre-tokenized audio codes. Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
            audio_output_codes_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            audio_input_embeds: Pre-computed audio input embeddings. Shape: [batch_size, num_frames, hidden_size].
                Dtype: float.
            audio_input_embeds_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            labels: Ground-truth token IDs for loss. Shape: [batch_size, seq_length]. Dtype: long.
            position_ids: Position indices. Shape: [batch_size, seq_length]. Dtype: long.
            past_key_values: KV cache for generation.
            inputs_embeds: Pre-computed input embeddings. Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            use_cache: Whether to return KV cache.
            cache_position: Position indices for static cache.
            use_speaker_embedding: Whether to compute speaker embeddings from `speaker_encoder_audio`.
            speaker_embeds: Pre-computed speaker embeddings. Shape: [num_speakers, 1, feature_dim]. Dtype: float.
            debug_mode: Enable debug forward pass.
            debug_step: Step index for debugging.
            **kwargs: Passed to text model.

        Returns:
            RaonModelOutput with loss, text_loss, audio_loss, aux_loss, hidden states, logits, and past_key_values.
        """
        self._validate_audio_output_inputs(
            audio_output=audio_output,
            audio_output_codes=audio_output_codes,
            audio_output_codes_mask=audio_output_codes_mask,
        )
        self._validate_audio_input_inputs(
            input_ids=input_ids,
            audio_input=audio_input,
            audio_input_embeds=audio_input_embeds,
            audio_input_embeds_mask=audio_input_embeds_mask,
        )
        speaker_encoder = self.speaker_encoder
        need_speaker_embedding = (
            audio_output is not None and use_speaker_embedding and speaker_embeds is None and speaker_encoder is not None
        )

        if self.supports_audio_output and audio_output_codes is None:
            with torch.no_grad():
                if audio_output_segments is not None and audio_output is not None:
                    # Utterance-segmented encoding: each speech segment is encoded
                    # independently so codes align with [A] positions (no SIL frames).
                    audio_output_inputs = self.tokenize_audio_segments(
                        audio=audio_output,
                        segments=audio_output_segments,
                        num_code_groups=self.num_code_groups,
                        return_mimi_features=False,
                    )
                else:
                    audio_output_inputs = self.tokenize_audio(
                        audio=audio_output,
                        audio_lengths=audio_output_lengths,
                        num_code_groups=self.num_code_groups,
                        return_mimi_features=False,
                    )
            audio_output_codes = audio_output_inputs.audio_codes
            audio_output_codes_mask = audio_output_inputs.audio_codes_mask

        if need_speaker_embedding:
            assert self.is_pretrained_speaker_encoder, (
                "Speaker embedding requires pretrained speaker encoder path "
                "with speaker_encoder_audio inputs."
            )
            if speaker_encoder_audio is not None:
                assert speaker_encoder is not None, "speaker_encoder is required when use_speaker_embedding is enabled."
                assert speaker_encoder_audio_lengths is not None, (
                    "speaker_encoder_audio_lengths is required when speaker_encoder_audio is provided."
                )
                assert speaker_encoder_audio.shape[0] == speaker_encoder_audio_lengths.shape[0], (
                    "speaker_encoder_audio and speaker_encoder_audio_lengths must have matching batch size. "
                    f"Got `{speaker_encoder_audio.shape[0]=}` and `{speaker_encoder_audio_lengths.shape[0]=}`."
                )
                speaker_embeds = speaker_encoder(speaker_encoder_audio, speaker_encoder_audio_lengths)

        # Speaker embedding dropout reduces reliance on the speaker-conditioning path.
        if self.training and speaker_embeds is not None and self.speaker_dropout > 0:
            drop_mask = torch.rand(speaker_embeds.shape[0], device=speaker_embeds.device) < self.speaker_dropout
            if drop_mask.all():
                speaker_embeds = None
            elif drop_mask.any():
                speaker_embeds = speaker_embeds.clone()
                speaker_embeds[drop_mask] = 0.0

        if self.supports_audio_input and audio_input_embeds is None and audio_input is not None:
            audio_input_outputs = self.get_audio_input_embeds(
                audio=audio_input,
                audio_lengths=audio_input_lengths,
            )
            audio_input_embeds = audio_input_outputs.audio_embeds
            audio_input_embeds_mask = audio_input_outputs.audio_embeds_mask

        if inputs_embeds is None:
            assert input_ids is not None, "input_ids is required when inputs_embeds is None."
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
            assert inputs_embeds is not None, "get_input_embeddings must return non-None."
            inputs_embeds = self.update_inputs_embeds(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                audio_output_codes=audio_output_codes,
                audio_output_codes_mask=audio_output_codes_mask,
                audio_input_embeds=audio_input_embeds,
                audio_input_embeds_mask=audio_input_embeds_mask,
                speaker_embeds=speaker_embeds,
            )
        else:
            assert input_ids is None and audio_output_codes is None and audio_input_embeds is None, (
                "When inputs_embeds is provided, input_ids, audio_output_codes, and audio_input_embeds must be None."
            )

        inputs_embeds = cast_to_module_dtype(inputs_embeds, self.text_model)
        if debug_mode:
            if input_ids is None or input_ids.numel() == 0:
                logger.info("forward debug latest token: input_ids is None or empty")
            else:
                if attention_mask is not None:
                    last_pos = attention_mask.long().sum(dim=1).clamp_min(1) - 1
                else:
                    last_pos = torch.full(
                        (input_ids.shape[0],),
                        input_ids.shape[1] - 1,
                        device=input_ids.device,
                        dtype=torch.long,
                    )

                batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
                last_token_ids = input_ids[batch_idx, last_pos].detach().cpu().tolist()
                last_pos_list = last_pos.detach().cpu().tolist()

                def _token_kind(token_id: int) -> str:
                    if token_id == AUDIO_INPUT_PLACEHOLDER.id:
                        return "audio_input_placeholder"
                    if token_id == AUDIO_OUTPUT_PLACEHOLDER.id:
                        return "audio_output_placeholder"
                    if token_id == AUDIO_OUTPUT_PAD.id:
                        return "audio_output_pad"
                    if token_id == AUDIO_OUTPUT_END_PAD.id:
                        return "audio_output_end_pad"
                    if token_id == AUDIO_OUTPUT_BC.id:
                        return "audio_output_backchannel"
                    if token_id == DUPLEX_SIL.id:
                        return "duplex_sil"
                    if token_id == AUDIO_START.id:
                        return "audio_start"
                    if token_id == IM_START.id:
                        return "im_start"
                    if token_id == SPEAKER_EMBEDDING_PLACEHOLDER.id:
                        return "speaker_embedding_placeholder"
                    if self.speaker_token_id is not None and token_id == self.speaker_token_id:
                        return "speaker_placeholder"
                    return "text_or_other"

                max_samples = min(4, len(last_token_ids))
                i = max_samples - 1
                token_id_i = int(last_token_ids[i])
                pos_i = int(last_pos_list[i])
                kind_i = _token_kind(token_id_i)
                token_view = self._decode_token_id_for_debug(token_id_i) if kind_i == "text_or_other" else kind_i
                logger.info(
                    "forward debug sample %d token %d: %s (id=%d)",
                    i,
                    pos_i,
                    token_view,
                    token_id_i,
                )

        text_outputs = self.text_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        assert self.accepted_thinker_hidden_states is not None, (
            "accepted_thinker_hidden_states must be set by thinker capture hook."
        )
        assert text_outputs.last_hidden_state is not None, "text model must return last_hidden_state."

        accepted_hidden = self.accepted_thinker_hidden_states
        self.accepted_thinker_hidden_states = None

        # Text logits from thinker output (post-norm from text_model.norm, like standard LLM).
        text_logits = self.lm_head(text_outputs.last_hidden_state)

        # Talker forward: project thinker hidden → talker → audio hidden states.
        if self.talker is not None and self.thinker_to_talker_proj is not None:
            talker_input = self.thinker_to_talker_proj(accepted_hidden)
            talker_cache = getattr(self, "_talker_past_key_values", None)
            talker_outputs = self.talker(
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=talker_input,
                past_key_values=talker_cache,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            if use_cache:
                self._talker_past_key_values = talker_outputs.past_key_values
            talker_last_hidden_state = talker_outputs.last_hidden_state
        else:
            # STT-only: no separate talker, use thinker output directly.
            talker_last_hidden_state = text_outputs.last_hidden_state

        loss = None
        text_loss = None
        audio_loss = None
        audio_logits = None
        if labels is not None:
            text_labels = self.get_text_labels(labels)
            text_loss = self.unreduced_causal_lm_loss(logits=text_logits, labels=text_labels)
            combined_loss, audio_loss, audio_logits = self.ddp_safe_loss(
                text_loss=text_loss,
                text_labels=text_labels,
                input_ids=input_ids,
                labels=labels,
                hidden_embeds=talker_last_hidden_state,
                audio_output_codes=audio_output_codes,
                audio_output_codes_mask=audio_output_codes_mask,
                speaker_embeds=speaker_embeds,
            )
            loss = combined_loss.mean()

        router_logits = getattr(text_outputs, "router_logits", None)
        aux_loss: torch.Tensor | None = None

        if self.output_losses_only:
            return RaonModelOutput(loss=loss, text_loss=text_loss, audio_loss=audio_loss)
        else:
            return RaonModelOutput(
                loss=loss,
                text_loss=text_loss,
                audio_loss=audio_loss,
                aux_loss=aux_loss,
                text_last_hidden_state=text_outputs.last_hidden_state,
                talker_last_hidden_state=talker_last_hidden_state,
                text_logits=text_logits,
                audio_logits=audio_logits,
                router_logits=router_logits,
                past_key_values=text_outputs.past_key_values,
            )

    def inference_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        audio_input: torch.Tensor | None = None,
        audio_output: torch.Tensor | None = None,
        audio_input_lengths: torch.Tensor | None = None,
        audio_output_lengths: torch.Tensor | None = None,
        audio_output_codes: torch.Tensor | None = None,
        audio_output_codes_mask: torch.Tensor | None = None,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        speaker_embeds: torch.Tensor | None = None,
        use_cache: bool | None = False,
        past_key_values: DynamicCache | StaticCache | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference forward pass and return talker hidden state and text logits for decoding.

        Args:
            input_ids: Token IDs. Shape: [batch_size, seq_length]. Dtype: long.
            attention_mask: Valid position mask. Shape: [batch_size, seq_length]. Dtype: long.
            position_ids: Position indices. Shape: [batch_size, seq_length]. Dtype: long.
            audio_input: Raw input audio. Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_output: Raw output audio. Same shape as audio_input.
            audio_input_lengths: Valid sample lengths. Shape: [batch_size]. Dtype: long.
            audio_output_lengths: Valid sample lengths. Shape: [batch_size]. Dtype: long.
            audio_output_codes: Pre-tokenized output codes. Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
            audio_output_codes_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            audio_input_embeds: Pre-computed audio input embeddings. Shape: [batch_size, num_frames, hidden_size].
                Dtype: float.
            audio_input_embeds_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            speaker_embeds: Speaker conditioning. Shape: [batch_size, num_frames, feature_dim]. Dtype: float.
            use_cache: Whether to use KV cache.
            past_key_values: KV cache for incremental decoding.
            cache_position: Explicit cache position indices. Shape: [seq_length]. Dtype: long.

        Returns:
            Tuple of (talker_last_hidden_state, text_logits). talker_last_hidden_state Shape: [batch_size, seq_length,
            hidden_size]. Dtype: float. text_logits Shape: [batch_size, seq_length, vocab_size]. Dtype: float.
        """
        outputs: RaonModelOutput = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            audio_input=audio_input,
            audio_output=audio_output,
            audio_input_lengths=audio_input_lengths,
            audio_output_lengths=audio_output_lengths,
            audio_output_codes=audio_output_codes,
            audio_output_codes_mask=audio_output_codes_mask,
            audio_input_embeds=audio_input_embeds,
            audio_input_embeds_mask=audio_input_embeds_mask,
            speaker_embeds=speaker_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        assert isinstance(talker_last_hidden_state := outputs.talker_last_hidden_state, torch.Tensor), (
            "forward must return talker_last_hidden_state as a tensor."
        )
        assert isinstance(text_logits := outputs.text_logits, torch.Tensor), "forward must return text_logits as a tensor."
        return talker_last_hidden_state, text_logits

    def generate_audio_codes(
        self,
        talker_last_hidden_state: torch.Tensor,
        first_code_sampler: Callable[[torch.Tensor], torch.Tensor] | None = None,
        allow_audio_end: bool = True,
    ) -> torch.Tensor:
        """Generate autoregressive audio codes from the last talker hidden state.

        Args:
            talker_last_hidden_state: Talker hidden states. Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            first_code_sampler: Optional callable to sample first code from logits; else argmax.
            allow_audio_end: If False, suppress AUDIO_END sampling for duplex-style decoding.

        Returns:
            Generated audio codes. Shape: [batch_size, num_generated_frames, num_code_groups]. Dtype: long.
        """
        assert self.audio_lm_head is not None, "audio_lm_head is unavailable when supports_audio_output is False."
        assert self.proj_code is not None, "proj_code is unavailable when supports_audio_output is False."
        assert self.code_predictor is not None, "code_predictor is unavailable when supports_audio_output is False."

        first_code_logits = self.audio_lm_head(talker_last_hidden_state[:, -1])

        if not allow_audio_end and first_code_logits.shape[-1] > self.codebook_size:
            # Suppress AUDIO_END token (only when audio_lm_head produces codebook_size+1 logits)
            first_code_logits = first_code_logits.clone()
            first_code_logits[..., self.codebook_size] = torch.finfo(first_code_logits.dtype).min

        if first_code_sampler is not None:
            first_code = first_code_sampler(first_code_logits)
        else:
            first_code = first_code_logits.argmax(dim=-1, keepdim=True)

        audio_end_mask = first_code[:, 0] == self.codebook_size
        safe_first_code = first_code.clamp_max(self.codebook_size - 1)

        hidden_embeds = self.proj_code(talker_last_hidden_state[:, -1:])
        inputs_embeds = torch.cat(
            (hidden_embeds, self.code_predictor.get_input_embeddings()(safe_first_code)),
            dim=1,
        )
        sequences = self.code_predictor.predict_codes(inputs_embeds=inputs_embeds)
        sequences = torch.cat((first_code, sequences.to(first_code.device)), dim=1)
        if audio_end_mask.any():
            sequences[audio_end_mask, 1:] = 0
        return sequences

    def decode_audio(
        self,
        audio_codes: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        use_streaming: bool | None = None,
    ) -> AudioDecoderOutput:
        """Decode discrete audio codes to waveform via the audio tokenizer decoder.

        Args:
            audio_codes: Discrete audio codes. Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
            padding_mask: Mask indicating valid audio sample positions for trimming decoder output.
                Shape: [batch_size, num_samples]. Dtype: long.
            use_streaming: Whether to use the streaming Mimi decoder. ``None`` falls back to tokenizer config.

        Returns:
            AudioDecoderOutput with audio waveform (Shape: [batch_size, num_samples]. Dtype: float.).
        """
        assert self.audio_tokenizer is not None, "audio_tokenizer is unavailable when supports_audio_output is False."
        outputs = self.audio_tokenizer.decode(
            audio_codes.transpose(1, 2),
            padding_mask=padding_mask,
            use_streaming=use_streaming,
            return_dict=True,
        )
        assert isinstance(outputs, StreamingMimiDecoderOutput), "tokenizer decode output must be StreamingMimiDecoderOutput."
        assert (audio_values := outputs.audio_values) is not None, "decode output must contain audio_values."
        audio = audio_values.view(audio_values.shape[0], audio_values.shape[2])
        return AudioDecoderOutput(audio=audio)

    def init_past_key_values(
        self,
        batch_size: int,
        max_sequence_length: int,
        prev_cache: Cache | None = None,
    ) -> Cache:
        """Initialize or reset KV cache for text model incremental decoding.

        When the model has a separate talker, also initializes a
        DynamicCache for the talker model stored as ``_talker_past_key_values``.

        Args:
            batch_size: Batch size for the cache.
            max_sequence_length: Maximum sequence length to cache.
            prev_cache: Existing cache to reset and reuse; if None, creates a new StaticCache.

        Returns:
            Initialized StaticCache ready for incremental decoding.
        """
        # Initialize talker KV cache for the separate HF talker model.
        if self.talker is not None:
            self._talker_past_key_values: DynamicCache | None = DynamicCache()

        if prev_cache is not None:
            prev_cache.reset()
            return prev_cache

        return StaticCache(
            self.config.text_model_config,
            max_cache_len=max_sequence_length,
        )

    def free_past_key_values(self, past_key_values: Cache) -> None:
        """Release KV cache resources including talker KV cache."""
        # Clean up talker KV cache.
        if hasattr(self, "_talker_past_key_values"):
            self._talker_past_key_values = None

    def _set_attention_implementation(self, attn_implementation: Literal["sdpa", "flash_attention_2"]) -> None:
        """Set attention implementation for text model and code predictor."""
        self.text_model.config._attn_implementation = attn_implementation  # type: ignore
        if self.code_predictor is not None:
            self.code_predictor.config._attn_implementation = attn_implementation  # type: ignore

    def _decode_token_id_for_debug(self, token_id: int) -> str:
        """Best-effort decode of a text token id for debug logs."""
        if self._debug_tokenizer is None and not self._debug_tokenizer_disabled:
            model_name_or_path = getattr(self.config, "_name_or_path", None)
            if isinstance(model_name_or_path, str) and model_name_or_path:
                try:
                    self._debug_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("debug tokenizer load failed for %s: %s", model_name_or_path, exc)
                    self._debug_tokenizer_disabled = True
            else:
                self._debug_tokenizer_disabled = True

        if self._debug_tokenizer is None:
            return f"<id:{token_id}>"

        try:
            tok = self._debug_tokenizer.convert_ids_to_tokens(int(token_id))
            if tok is None:
                return f"<id:{token_id}>"
            return str(tok)
        except Exception as exc:  # noqa: BLE001
            logger.debug("debug token decode failed for id=%d: %s", token_id, exc)
            return f"<id:{token_id}>"


# Duplex config — supports checkpoints with model_type="raon_duplex"
class RaonDuplexConfig(RaonConfig):
    """Configuration alias for full-duplex checkpoints (model_type='raon_duplex')."""

    model_type = "raon_duplex"

    def __init__(self, **kwargs: Any) -> None:
        # Duplex-specific defaults; overridden by values in config.json when present.
        kwargs.setdefault("sequence_mode", "uta")
        kwargs.setdefault("use_sil_token", True)
        kwargs.setdefault("no_audio_in_sil", False)
        kwargs.setdefault("text_lookahead", 0)
        kwargs.setdefault("use_duplex_end_pad", True)
        kwargs.setdefault("duplex_pad_token_id", 151677)
        kwargs.setdefault("duplex_end_pad_token_id", 151678)
        kwargs.setdefault("duplex_sil_token_id", 151672)
        kwargs.setdefault("duplex_bc_token_id", 151673)
        kwargs.setdefault("speaker_token_id", 151671)
        kwargs.setdefault("audio_input_token_id", 151676)
        kwargs.setdefault("audio_output_token_id", 151675)
        kwargs.setdefault("audio_start_token_id", 151669)
        kwargs.setdefault("im_start_token_id", 151644)
        # Loss weight defaults (overridable at training time via duplex_train args)
        kwargs.setdefault("text_loss_weight", 1.0)
        kwargs.setdefault("sil_loss_weight", 1.0)
        kwargs.setdefault("epad_loss_weight", 0.0)
        kwargs.setdefault("semantic_loss_weight", 1.0)
        kwargs.setdefault("acoustic_loss_weights", None)
        super().__init__(**kwargs)


# Duplex model — same architecture, different model_type for HF registry
class RaonDuplexModel(RaonModel):
    """Model alias for full-duplex checkpoints (model_type='raon_duplex')."""

    config_class = RaonDuplexConfig


# Register model types with HuggingFace AutoConfig/AutoModel
from transformers import AutoConfig, AutoModel  # noqa: E402

# Method 2: register so `import raon` makes AutoModel.from_pretrained() work
AutoConfig.register("raon", RaonConfig)
AutoModel.register(RaonConfig, RaonModel)

AutoConfig.register("raon_duplex", RaonDuplexConfig)
AutoModel.register(RaonDuplexConfig, RaonDuplexModel)
