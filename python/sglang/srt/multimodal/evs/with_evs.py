import dataclasses
import typing
from dataclasses import dataclass, fields
from itertools import chain, pairwise
from typing import Callable

import torch
from transformers import PretrainedConfig

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult
from sglang.srt.multimodal.evs.evs_core import (
    compute_retained_tokens_count,
    compute_retention_mask,
)
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils.common import get_bool_env_var
from sglang.utils import logger


@dataclasses.dataclass(kw_only=True)
class VideoEVSDataItem(MultimodalDataItem):
    modality: Modality = Modality.VIDEO
    frames_per_video: list[int]

    def __post_init__(self):
        assert self.is_video()


@dataclass(frozen=True, kw_only=True)
class EVSConfig:
    video_pruning_rate: float
    downsample_ratio: float
    patch_size: int
    force_image_size: int
    num_image_token: int

    @classmethod
    def from_pretrained_config(cls, config: PretrainedConfig) -> "EVSConfig":
        config_cls_name = type(config).__name__
        assert isinstance(
            config, PretrainedConfig
        ), f"Expected Model config to be a PretrainedConfig, got {config_cls_name}"
        field_names = [field.name for field in fields(cls)]
        for key in field_names:
            if not hasattr(config, key):
                raise AttributeError(
                    f"Expected Model config of type {config_cls_name} to have a `{key}` attribute"
                )
        return EVSConfig(**{key: getattr(config, key) for key in field_names})


def evs_rerender_pruned_frames(
    input_ids: torch.Tensor,
    *,
    frame_offsets: list[tuple[int, int]],
    num_tokens_per_frame: list[int],
) -> torch.Tensor:
    assert len(frame_offsets) == len(
        num_tokens_per_frame
    ), "Number of frame offsets must match number of tokens per frame"
    filler_token_id = input_ids[frame_offsets[0][0]]
    offsets_set = set(frame_offsets)
    final = []
    frame_idx = 0
    all_pairs = list(
        pairwise(sorted({-1, *chain.from_iterable(offsets_set), len(input_ids)}))
    )
    for i, (start, end) in enumerate(all_pairs):
        if (start, end) in offsets_set:
            num_tokens = num_tokens_per_frame[frame_idx]
            final.append(
                torch.tensor(
                    [filler_token_id] * num_tokens,
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
            )
            frame_idx += 1
        else:
            if i + 1 < len(all_pairs) and num_tokens_per_frame[i + 1] == 0:
                continue  # if the next frame is empty, don't render its timestamp at all
            final.append(input_ids[start + 1 : end])
    final_tensor = torch.cat(final)
    assert len(final_tensor) == len(
        input_ids
    ), "Number of final tokens must match number of input tokens"
    return final_tensor


@dataclass(kw_only=True)
class EVSEmbeddingResult(EmbeddingResult):
    num_tokens_per_frame: list[int]


class EVS:
    def __init__(
        self,
        get_video_feature: typing.Callable[[list[MultimodalDataItem]], torch.Tensor],
        *,
        config: EVSConfig,
        model_name: str,
        verbose: bool = False,
    ) -> None:
        self.get_video_feature = get_video_feature
        self.model_name = model_name

        self.q = config.video_pruning_rate
        assert self.q > 0.0 and self.q < 1.0

        self.num_tokens_per_frame = config.num_image_token
        self.rows = self.cols = int(config.num_image_token**0.5)

        verbose = get_bool_env_var(
            "SGLANG_EVS_VERBOSE", default="false" if not verbose else "true"
        )
        self.logger = logger.info if verbose else logger.debug

    def __call__(self, items: list[MultimodalDataItem]) -> EVSEmbeddingResult:
        self.logger(f"Beginning EVS for model {self.model_name}")
        if len(items) > 1:
            raise MultimodalDataItem.MultimodalDataItemContainsAllItemsOfSameModality
        item = items[0]
        assert isinstance(item, VideoEVSDataItem)
        videos_features = self.get_video_feature([item])

        final_embeddings: list[torch.Tensor] = []
        num_tokens_per_frame: list[int] = []

        for single_video in videos_features.split(item.frames_per_video):
            num_frames = single_video.shape[0]

            video_size_thw = (num_frames, self.rows, self.cols)

            retention_mask = compute_retention_mask(
                single_video,
                video_size_thw=video_size_thw,
                spatial_merge_size=1,
                q=self.q,
            ).view(num_frames, -1)

            preserved = single_video[retention_mask]
            final_embeddings.append(preserved)
            retention_mask_thw = retention_mask.reshape(video_size_thw)
            num_tokens_per_frame.extend(
                retention_mask_thw.sum(dim=(1, 2)).long().tolist()
            )
        final_embeddings_tensor = torch.cat(final_embeddings)
        return EVSEmbeddingResult(
            embedding=final_embeddings_tensor,
            num_tokens_per_frame=num_tokens_per_frame,
        )


CreateEVSConfig = Callable[[PretrainedConfig], EVSConfig]


class VideoModel(typing.Protocol):
    def get_video_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor: ...


_CREATE_EVS_CONFIG: dict[type[VideoModel], CreateEVSConfig] = {}


VideoModelType = typing.TypeVar("VideoModelType", bound=VideoModel)


def with_EVS(
    model_cls: type[VideoModelType],
    *,
    create_evs_config: CreateEVSConfig = EVSConfig.from_pretrained_config,
) -> type[VideoModelType]:
    model_name = model_cls.__name__

    if not hasattr(model_cls, "get_video_feature"):
        raise AttributeError(
            f"Expected Model {model_name} to have a `get_video_feature` method due to it being decorated with @with_EVS"
        )

    original_init = model_cls.__init__

    def __init__(
        self: VideoModelType,
        config: PretrainedConfig,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        evs_config = create_evs_config(config)
        self.evs_config = evs_config
        if evs_config.video_pruning_rate <= 0.0:
            logger.warning(
                f"EVS was requested on model {model_name} but is disabled for pruning_rate <= 0.0 ({evs_config.video_pruning_rate=}). EVS will be disabled"
            )
            self.evs_config = None
        else:
            logger.info(
                f"EVS will be enabled for model {model_name} [video_pruning_rate={evs_config.video_pruning_rate}]"
            )
            self.get_video_feature = EVS(
                self.get_video_feature, config=evs_config, model_name=model_name
            )
        original_init(self, config, *args, **kwargs)

    model_cls.__init__ = __init__
    _CREATE_EVS_CONFIG[model_cls] = create_evs_config
    return model_cls


def evs_tokens_per_frame(
    config: EVSConfig,
    num_frames: int,
) -> list[int]:
    retained = compute_retained_tokens_count(
        tokens_per_frame=config.num_image_token,
        num_frames=num_frames,
        q=config.video_pruning_rate,
    )
    base = retained // num_frames
    rem = retained % num_frames
    return [base] * (num_frames - 1) + [base + rem]


def resolve_evs_config(processor: BaseMultimodalProcessor) -> EVSConfig | None:
    models_with_evs = {
        _CREATE_EVS_CONFIG.get(model, None) for model in processor.models
    }
    assert (
        len(models_with_evs) == 1
    ), "All models must have EVS enabled or disabled, with the same create_evs_config function"
    create_evs_config = models_with_evs.pop()
    if create_evs_config is not None:
        cfg = create_evs_config(processor.hf_config)
        if cfg.video_pruning_rate > 0.0:
            return cfg
    return None
