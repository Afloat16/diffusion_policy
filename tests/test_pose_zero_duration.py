import numpy as np
import pytest
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

@pytest.mark.parametrize("finite_speed", [False, True])
def test_drive_repeated_pose_at_current_time(finite_speed):
    pose = np.zeros(6)
    interp = PoseTrajectoryInterpolator([0., 1.], np.zeros((2, 6)))
    speed = 1. if finite_speed else np.inf
    result = interp.drive_to_waypoint(pose, time=.5, curr_time=.5,
                                     max_pos_speed=speed, max_rot_speed=speed)
    np.testing.assert_array_equal(result.times, [.5])
    np.testing.assert_array_equal(result(np.array([-.5, .5, 1.5])), np.zeros((3, 6)))


def test_instantaneous_drive_preserves_requested_pose():
    pose = np.array([2., 3., 4., .1, .2, .3])
    interp = PoseTrajectoryInterpolator([0.], np.zeros((1, 6)))
    result = interp.drive_to_waypoint(pose, time=0., curr_time=0.)
    np.testing.assert_array_equal(result(0.), pose)


def test_schedule_existing_endpoint_is_idempotent():
    poses = np.array([[0., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.]])
    interp = PoseTrajectoryInterpolator([0., 1.], poses)
    result = interp.schedule_waypoint(poses[-1], time=1., max_pos_speed=1., max_rot_speed=1.)
    np.testing.assert_array_equal(result.times, [0., 1.])
    np.testing.assert_allclose(result(np.linspace(0., 1., 7)), interp(np.linspace(0., 1., 7)))


def test_drive_nonzero_distance_still_respects_speed():
    interp = PoseTrajectoryInterpolator([0.], np.zeros((1, 6)))
    pose = np.array([2., 0., 0., 0., 0., .5])
    result = interp.drive_to_waypoint(pose, time=0., curr_time=0., max_pos_speed=1., max_rot_speed=.5)
    np.testing.assert_allclose(result.times, [0., 2.])
    np.testing.assert_allclose(result(1.)[:3], [1., 0., 0.])
    np.testing.assert_allclose(result(2.), pose)
