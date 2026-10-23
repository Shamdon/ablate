from types import SimpleNamespace

import torch

from ablate.extract import collect_activations


class _FakeModel:
    def __call__(self, input_ids, attention_mask, output_hidden_states):
        activations = input_ids.float().unsqueeze(-1)
        return SimpleNamespace(hidden_states=(activations, activations))


class _FakeLM:
    model = _FakeModel()

    def tokenize(self, prompts, system=None):
        # Left padding: the first prompt has one real token and the second has
        # three. The padding token deliberately has a different activation.
        return {
            "input_ids": torch.tensor([[0, 0, 5], [7, 8, 9]]),
            "attention_mask": torch.tensor([[0, 0, 1], [1, 1, 1]]),
        }


def test_collect_activations_averages_only_real_trailing_tokens():
    activations = collect_activations(
        _FakeLM(), ["short", "long"], positions=2, batch_size=2
    )

    assert activations.shape == (2, 2, 1)
    assert activations[:, 0, 0].tolist() == [5.0, 5.0]
    assert activations[:, 1, 0].tolist() == [8.5, 8.5]


def test_collect_activations_with_single_position_uses_last_real_token():
    activations = collect_activations(
        _FakeLM(), ["short", "long"], positions=1, batch_size=2
    )

    assert activations[:, 0, 0].tolist() == [5.0, 5.0]
    assert activations[:, 1, 0].tolist() == [9.0, 9.0]
