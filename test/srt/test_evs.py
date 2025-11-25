from dataclasses import asdict, dataclass

import pytest


def test_resolve_evs_config():
    from sglang.srt.multimodal.evs.evs_mixin import (
        EVSConfig,
        EVSMixin,
        resolve_evs_config,
    )

    evs_config = EVSConfig(video_pruning_rate=0.0, full_frame_num_tokens=256)

    class FakeEVSModel(EVSMixin):
        @classmethod
        def create_evs_config(cls, config):
            return evs_config

    class FakeEVSConfig:
        model_type = FakeEVSModel.__name__

    class NonEVSModel:
        pass

    class NonEVSConfig:
        model_type = NonEVSModel.__name__

    class FakeProcessor:
        models = [FakeEVSModel, NonEVSModel]

        def __init__(self, hf_config):
            self.hf_config = hf_config

    processor = FakeProcessor(FakeEVSConfig())
    assert asdict(resolve_evs_config(processor)) == asdict(evs_config)

    processor = FakeProcessor(NonEVSConfig())
    assert resolve_evs_config(processor) is None


@dataclass(kw_only=True)
class Case:
    input_ids: list[int]
    frame_offsets_inclusive: list[tuple[int, int]]
    num_tokens_per_frame: list[int]
    expected_output_ids: list[int]


FILL = 0

# fmt: off
@pytest.mark.parametrize("case", [
    Case(
        input_ids=[1, FILL, FILL, 4, 5, FILL, FILL, FILL, 9, 10, FILL, FILL, 12, 13],
        expected_output_ids=[1, FILL, 4, 5, FILL, FILL, FILL, FILL, 9, 10, FILL, FILL, 12, 13],
        frame_offsets_inclusive=[(1, 2), (5, 7), (10, 11)],
        num_tokens_per_frame=[1, 4, 2],
    ),
    Case(
        input_ids=[1, FILL, FILL, 4, 5, 9, 10, FILL, FILL, FILL],
        expected_output_ids=[1, FILL, 4, 5, 9, 10, FILL, FILL, FILL, FILL],
        frame_offsets_inclusive=[(1, 2), (7, 9)],
        num_tokens_per_frame=[1, 4],
    ),
    Case(
        input_ids=[FILL, FILL, 1, 4, FILL, FILL, FILL, 5, 9, 10],
        expected_output_ids=[FILL, 1, 4, FILL, FILL, FILL, FILL, 5, 9, 10],
        frame_offsets_inclusive=[(0, 1), (4, 6)],
        num_tokens_per_frame=[1, 4],
    ),
]) # fmt: off
def test_redistribute_placeholder_tokens_by_tokens_per_frame(case: Case):
    import torch

    from sglang.srt.multimodal.evs.evs_core import (
        redistribute_placeholder_tokens_by_tokens_per_frame,
    )

    output_ids = redistribute_placeholder_tokens_by_tokens_per_frame(input_ids=torch.tensor(case.input_ids), frame_offsets_inclusive=case.frame_offsets_inclusive, num_tokens_per_frame=case.num_tokens_per_frame)
    assert output_ids.tolist() == case.expected_output_ids

if __name__ == "__main__":
    pytest.main([__file__])
