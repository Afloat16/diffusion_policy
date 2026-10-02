"""Appending invalid episodes must preserve the replay buffer."""
import unittest
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer


class ReplayAppendValidationTest(unittest.TestCase):
    def make_buffer(self, backend):
        if backend == "numpy":
            buffer = ReplayBuffer.create_empty_numpy()
        else:
            buffer = ReplayBuffer.create_empty_zarr()
        buffer.add_episode({
            "a": np.array([[1., 2.], [3., 4.]]),
            "b": np.array([[5.], [6.]]),
        })
        return buffer

    def assert_unchanged(self, buffer, before):
        np.testing.assert_array_equal(buffer.episode_ends[:], [2])
        self.assertEqual(buffer.n_steps, 2)
        for key, expected in before.items():
            np.testing.assert_array_equal(buffer[key][:], expected)

    def test_late_feature_error_does_not_resize_earlier_fields(self):
        for backend in ("numpy", "zarr"):
            with self.subTest(backend=backend):
                buffer = self.make_buffer(backend)
                before = {k: v[:].copy() for k, v in buffer.items()}
                with self.assertRaises(ValueError):
                    buffer.add_episode({"a": np.ones((1, 2)), "b": np.ones((1, 2))})
                self.assert_unchanged(buffer, before)

    def test_missing_field_does_not_corrupt_episode_metadata(self):
        for backend in ("numpy", "zarr"):
            with self.subTest(backend=backend):
                buffer = self.make_buffer(backend)
                before = {k: v[:].copy() for k, v in buffer.items()}
                with self.assertRaises(ValueError):
                    buffer.add_episode({"a": np.ones((1, 2))})
                self.assert_unchanged(buffer, before)

    def test_empty_and_mismatched_time_dimensions_leave_data_unchanged(self):
        for backend in ("numpy", "zarr"):
            for data in ({}, {"a": np.empty((0, 2)), "b": np.empty((0, 1))},
                         {"a": np.ones((1, 2)), "b": np.ones((2, 1))}):
                with self.subTest(backend=backend, shapes={k: v.shape for k, v in data.items()}):
                    buffer = self.make_buffer(backend)
                    before = {k: v[:].copy() for k, v in buffer.items()}
                    with self.assertRaises(ValueError):
                        buffer.add_episode(data)
                    self.assert_unchanged(buffer, before)

    def test_valid_append_still_allows_a_new_field(self):
        for backend in ("numpy", "zarr"):
            with self.subTest(backend=backend):
                buffer = self.make_buffer(backend)
                buffer.add_episode({"a": np.array([[7., 8.]]), "b": np.array([[9.]]),
                                    "c": np.array([[10.]])})
                np.testing.assert_array_equal(buffer.episode_ends[:], [2, 3])
                np.testing.assert_array_equal(buffer["c"][:], [[0.], [0.], [10.]])
                self.assertTrue(all(value.shape[0] == 3 for value in buffer.values()))


if __name__ == "__main__":
    unittest.main()

