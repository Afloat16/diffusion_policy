import numpy as np
import pytest
from diffusion_policy.common.replay_buffer import ReplayBuffer

@pytest.fixture(params=["numpy", "zarr"])
def buffer(request):
    obj = (ReplayBuffer.create_empty_numpy() if request.param == "numpy" else
           ReplayBuffer.create_empty_zarr())
    obj.add_episode({"obs": np.arange(6).reshape(3, 2), "action": np.arange(3).reshape(3, 1)})
    return obj

@pytest.mark.parametrize("bad", ["missing", "shape", "empty"])
def test_invalid_episode_does_not_mutate_buffer(buffer, bad):
    before = {k: np.asarray(v).copy() for k, v in buffer.items()}
    ends = np.asarray(buffer.episode_ends).copy()
    if bad == "missing":
        data = {"obs": np.ones((2, 2), dtype=np.int64)}
    elif bad == "shape":
        data = {"obs": np.ones((2, 2), dtype=np.int64), "action": np.ones((2, 4), dtype=np.int64)}
    else:
        data = {"obs": np.empty((0, 2), dtype=np.int64), "action": np.empty((0, 1), dtype=np.int64)}
    with pytest.raises(ValueError):
        buffer.add_episode(data)
    np.testing.assert_array_equal(buffer.episode_ends, ends)
    for key, value in before.items():
        np.testing.assert_array_equal(buffer[key], value)
    assert all(value.shape[0] == buffer.n_steps for value in buffer.values())


def test_new_fields_keep_zero_filled_history(buffer):
    buffer.add_episode({"obs": np.ones((2, 2), dtype=np.int64),
                        "action": np.ones((2, 1), dtype=np.int64),
                        "force": np.full((2, 3), 7., dtype=np.float32)})
    np.testing.assert_array_equal(buffer["force"][:3], np.zeros((3, 3)))
    np.testing.assert_array_equal(buffer["force"][3:], np.full((2, 3), 7.))
    np.testing.assert_array_equal(buffer.episode_lengths, [3, 2])
