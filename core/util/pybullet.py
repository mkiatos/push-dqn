"""
PyBullet Utilities
==================

"""

import pybullet as p
import numpy as np
from clutter.util.orientation import Quaternion


def get_joint_indices(names, body_unique_id=0):
    indices = []
    for i in range(p.getNumJoints(body_unique_id)):
        joint = p.getJointInfo(body_unique_id, i)
        for j in range(len(names)):
            if str(joint[1], 'utf-8') == names[j]:
                indices.append(joint[0])
                break
    return indices


def get_link_indices(names, body_unique_id=0):
    indices = []
    for i in range(p.getNumJoints(body_unique_id)):
        joint = p.getJointInfo(body_unique_id, i)
        for j in range(len(names)):
            if joint[12].decode('utf-8') == names[j]:
                indices.append(joint[0])
    return indices


def get_link_pose(name, body_unique_id=0):
    index = get_link_indices([name])[0]
    state = p.getLinkState(0, index)
    pos = [state[0][0], state[0][1], state[0][2]]
    q = state[1]
    quat = Quaternion(w=q[3], x=q[0], y=q[1], z=q[2])
    return pos, quat


def get_camera_pose(pos, target_pos, up_vector):
    """
    Computes the camera pose w.r.t. world

    Parameters
    ----------
    pos: np.array([3, 1])
        The position of the camera in world coordinates
    target_pos: np.array([3, 1])
        The position of the focus point in world coordinates e.g. lookAt vector
    up_vector: np.array([3, 1])
        Up vector of the camera in world coordinates

    Returns
    -------
    np.array([4, 4])
        The camera pose w.r.t. world frame
    """
    # Compute z axis
    z = target_pos - pos
    z /= np.linalg.norm(z)

    # Compute y axis. Project the up vector to the plane defined
    # by the point pos and unit vector z
    dist = np.dot(z, -up_vector)
    y = -up_vector - dist * z
    y /= np.linalg.norm(y)

    # Compute x axis as the cross product of y and z
    x = np.cross(y, z)
    x /= np.linalg.norm(x)

    camera_pose = np.eye(4)
    camera_pose[0:3, 0] = x
    camera_pose[0:3, 1] = y
    camera_pose[0:3, 2] = z
    camera_pose[0:3, 3] = pos

    return pos, Quaternion.from_rotation_matrix(camera_pose[0:3, 0:3])
