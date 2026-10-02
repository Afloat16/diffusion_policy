import numpy as np
import pytest
from diffusion_policy.common.replay_buffer import ReplayBuffer

@pytest.fixture(params=["numpy", "zarr"])
def buffer(request):
    obj = (ReplayBuffer.create_empty_numpy() if request.param == "numpy" else
           ReplayBuffer.create_empty_zarr())
    for start, end in [(0, 3), (3, 8), (8, 10)]:
        obj.add_episode({"obs": np.arange(start, end)[:, None]})
    return obj

@pytest.mark.parametrize("index", [-3, -2, -1, 0, 1, 2])
def test_episode_slice_matches_episode_selection(buffer, index):
    selected = buffer["obs"][buffer.get_episode_slice(index)]
    np.testing.assert_array_equal(selected, buffer.get_episode(index)["obs"])
    assert len(selected) == buffer.episode_lengths[index]

@pytest.mark.parametrize("index", [-4, 3])
def test_episode_slice_rejects_out_of_range_indices(buffer, index):
    with pytest.raises(IndexError):
        buffer.get_episode_slice(index)
