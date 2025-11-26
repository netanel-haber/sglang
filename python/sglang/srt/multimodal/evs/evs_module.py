import dataclasses
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult
from sglang.srt.multimodal.evs.evs_core import (
    compute_retained_tokens_count,
    compute_retention_mask,
)
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.utils import logger


@dataclasses.dataclass(kw_only=True)
class VideoEVSDataItem(MultimodalDataItem):
    modality: Modality = Modality.VIDEO
    frames_per_video: list[int]

    def __post_init__(self):
        assert self.is_video()


@dataclass(kw_only=True)
class EVSEmbeddingResult(EmbeddingResult):
    num_tokens_per_frame: list[int]


@dataclass(frozen=True, kw_only=True)
class EVSConfig:
    video_pruning_rate: float
    full_frame_num_tokens: int

    def __post_init__(self):
        assert (
            self.video_pruning_rate >= 0.0 and self.video_pruning_rate < 1.0
        ), f"Video pruning rate must be between 0.0 and 1.0, got {self.video_pruning_rate=}"
        assert self.full_frame_num_tokens > 0

    def tokens_per_frame(self, num_frames: int) -> list[int]:
        retained = compute_retained_tokens_count(
            tokens_per_frame=self.full_frame_num_tokens,
            num_frames=num_frames,
            q=self.video_pruning_rate,
        )
        base = retained // num_frames
        rem = retained % num_frames
        return [base] * (num_frames - 1) + [base + rem]


class EVSModule(torch.nn.Module, ABC):
    @staticmethod
    @abstractmethod
    def create_evs_config(config: PretrainedConfig) -> EVSConfig:
        raise NotImplementedError

    def __init__(
        self,
        config: PretrainedConfig,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__()
        model_name = self.__class__.__name__
        if not hasattr(self, "get_video_feature"):
            raise AttributeError(
                f"Expected Model {model_name} to have a `get_video_feature` method"
            )
        self.evs_config = self.create_evs_config(config)
        video_pruning_rate = self.evs_config.video_pruning_rate
        self.original_get_video_feature = self.get_video_feature
        if video_pruning_rate > 0.0:
            logger.info(f"[EVS] enabled for {model_name} [{video_pruning_rate=}]")
            self.get_video_feature = self.evs
        else:
            logger.info(
                f"[EVS] requested on model {model_name} but is disabled for pruning_rate == 0.0."
            )

    def evs(self, items: list[MultimodalDataItem]) -> EVSEmbeddingResult:
        q = self.evs_config.video_pruning_rate
        logger.debug(f"[EVS] beginning for model {self.__class__.__name__} [{q=}]")
        rows = cols = int(self.evs_config.full_frame_num_tokens**0.5)
        if len(items) > 1:
            raise MultimodalDataItem.MultimodalDataItemContainsAllItemsOfSameModality
        item = items[0]
        assert isinstance(item, VideoEVSDataItem)
        videos_features = self.original_get_video_feature([item])

        final_embeddings: list[torch.Tensor] = []
        num_tokens_per_frame: list[int] = []

        for single_video in videos_features.split(item.frames_per_video):
            num_frames = single_video.shape[0]

            video_size_thw = (num_frames, rows, cols)

            retention_mask = compute_retention_mask(
                single_video,
                video_size_thw=video_size_thw,
                spatial_merge_size=1,
                q=q,
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


@dataclass(frozen=True, kw_only=True)
class NonEVSConfig:
    frame_num_tokens: int


class EVSProcessor(BaseMultimodalProcessor, ABC):
    @staticmethod
    @abstractmethod
    def create_non_evs_config(hf_config: PretrainedConfig) -> NonEVSConfig:
        """account for models in processor.models that don't support EVS"""
        raise NotImplementedError

    def __init__(self, hf_config, *args, **kwargs):
        super().__init__(hf_config, *args, **kwargs)
        self.non_evs_config = self.create_non_evs_config(hf_config)

        config_name = hf_config.__class__.__name__
        processor_name = self.__class__.__name__
        model_name = hf_config.model_type

        assert isinstance(model_name, str)

        evs_models = {
            model.__name__: model
            for model in self.models
            if issubclass(model, EVSModule)
        }

        if len(evs_models) == 0:
            logger.warning(
                f"[EVS] No EVS models found for processor.models={self.models}"
            )

        identity = f"processor={processor_name} model={model_name} config={config_name}"
        self.evs_config = None
        if model_name in evs_models:
            evs_model = evs_models[model_name]
            evs_config = evs_model.create_evs_config(hf_config)
            assert (
                self.non_evs_config.frame_num_tokens == evs_config.full_frame_num_tokens
            )
            logger.info(f"[EVS] {evs_config} resolved for triplet {identity}")
            if evs_config.video_pruning_rate > 0.0:
                self.evs_config = evs_config
        else:
            logger.info(f"[EVS] no config found for triplet {identity}")

    def tokens_per_frame(self, num_frames: int) -> list[int]:
        if self.evs_config is not None:
            return self.evs_config.tokens_per_frame(num_frames)
        else:
            return [self.non_evs_config.frame_num_tokens] * num_frames

    def data_item(
        self,
        frames_per_video: list[int],
        *,
        feature: torch.Tensor,
        offsets: list[tuple[int, int]],
    ) -> MultimodalDataItem:
        if self.evs_config is not None:
            return VideoEVSDataItem(
                frames_per_video=frames_per_video, feature=feature, offsets=offsets
            )
        else:
            return MultimodalDataItem(
                modality=Modality.VIDEO, feature=feature, offsets=offsets
            )
