#!/usr/bin/env python3

import numpy as np
import numpy.matlib as mat
import numpy.linalg as lin
import math

import logging
logger = logging.getLogger('robamine.utils.orientation')

def quat2rot(q, shape="wxyz"):
    """
    Transforms a quaternion to a rotation matrix.
    """
    if shape == "wxyz":
        n  = q[0]
        ex = q[1]
        ey = q[2]
        ez = q[3]
    elif shape == "xyzw":
        n  = q[3]
        ex = q[0]
        ey = q[1]
        ez = q[2]
    else:
        raise RuntimeError("The shape of quaternion should be wxyz or xyzw. Given " + shape + " instead")

    R = mat.eye(3)

    R[0, 0] = 2 * (n * n + ex * ex) - 1
    R[0, 1] = 2 * (ex * ey - n * ez)
    R[0, 2] = 2 * (ex * ez + n * ey)

    R[1, 0] = 2 * (ex * ey + n * ez)
    R[1, 1] = 2 * (n * n + ey * ey) - 1
    R[1, 2] = 2 * (ey * ez - n * ex)

    R[2, 0] = 2 * (ex * ez - n * ey)
    R[2, 1] = 2 * (ey * ez + n * ex)
    R[2, 2] = 2 * (n * n + ez * ez) - 1

    return R;

def rot2quat(R, shape="wxyz"):
    """
    Transforms a rotation matrix to a quaternion.
    """

    q = [None] * 4

    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S=4*qwh
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S

    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
      S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
      q[0] = (R[2, 1] - R[1, 2]) / S
      q[1] = 0.25 * S
      q[2] = (R[0, 1] + R[1, 0]) / S
      q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
      S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
      q[0] = (R[0, 2] - R[2, 0]) / S
      q[1] = (R[0, 1] + R[1, 0]) / S
      q[2] = 0.25 * S
      q[3] = (R[1, 2] + R[2, 1]) / S
    else:
      S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
      q[0] = (R[1, 0] - R[0, 1]) / S
      q[1] = (R[0, 2] + R[2, 0]) / S
      q[2] = (R[1, 2] + R[2, 1]) / S
      q[3] = 0.25 * S

    return q / lin.norm(q);

def get_homogeneous_transformation(pose):
    """
    Returns a homogeneous transformation given a pose [position, quaternion]
    """
    M = mat.zeros((4, 4))
    p = pose[0:3]
    R = quat2rot(pose[3:7])
    for i in range(0, 3):
        M[i, 3] = p[i]
        for j in range(0, 3):
            M[i, j] = R[i, j]
    M[3, 3] = 1
    return M

def get_pose_from_homog(M):
    """
    Returns a pose [position, quaternion] from a homogeneous matrix
    """
    p = [None] * 3
    R = mat.eye(3)

    for i in range(0, 3):
        p[i] = M[i, 3]
        for j in range(0, 3):
            R[i, j] = M[i, j]

    q = rot2quat(R)
    return np.concatenate((p, q))

def skew_symmetric(vector):
    output = np.zeros((3, 3))
    output[0, 1] = -vector[2]
    output[0, 2] =  vector[1]
    output[1, 0] =  vector[2]
    output[1, 2] = -vector[0]
    output[2, 0] = -vector[1]
    output[2, 1] =  vector[0]
    return output

def screw_transformation(position, orientation):
    output = np.zeros((6, 6))
    output[0:3, 0:3] = orientation
    output[3:6, 3:6] = orientation
    output[3:6, 0:3] = np.matmul(skew_symmetric(position), orientation)
    return output

def rotation_6x6(orientation):
    output = np.zeros((6, 6))
    output[0:3, 0:3] = orientation
    output[3:6, 3:6] = orientation
    return output

def rot2angleaxis(R):
    angle = np.arccos((np.trace(R) - 1) / 2)
    if angle == 0:
        logger.warn('Angle is zero (the rotation identity)')
        axis = None
    else:
        axis = (1 / (2 * np.sin(angle))) * np.array([R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]])
        if np.linalg.norm(axis) == 0:
            logger.warn('Axis is zero (the rotation is around an axis exactly at pi)')
            axis = None
        else:
            axis = axis / np.linalg.norm(axis)
    return angle, axis

def angleaxis2rot(angle, axis):
    c = math.cos(angle)
    s = math.sin(angle)
    v = 1 - c
    kx = axis[0]
    ky = axis[1]
    kz = axis[2]

    R = np.eye(3)
    R[0, 0] = pow(kx, 2) * v + c
    R[0, 1] = kx * ky * v - kz * s
    R[0, 2] = kx * kz * v + ky * s

    R[1, 0] = kx * ky * v + kz * s
    R[1, 1] = pow(ky, 2) * v + c
    R[1, 2] = ky * kz * v - kx * s

    R[2, 0] = kx * kz * v - ky * s
    R[2, 1] = ky * kz * v + kx * s
    R[2, 2] = pow(kz, 2) * v + c

    return R

class Affine3:
    def __init__(self):
        self.linear = np.eye(3)
        self.translation = np.zeros(3)

    @classmethod
    def from_matrix(cls, matrix):
        assert isinstance(matrix, np.ndarray)
        result = cls()
        result.translation = matrix[:3, 3]
        result.linear = matrix[0:3, 0:3]
        return result

    @classmethod
    def from_vec_quat(cls, pos, quat):
        assert isinstance(pos, np.ndarray)
        assert isinstance(quat, Quaternion)
        result = cls()
        result.translation = pos.copy()
        result.linear = quat.rotation_matrix()
        return result

    def matrix(self):
        matrix = np.eye(4)
        matrix[0:3, 0:3] = self.linear
        matrix[0:3, 3] = self.translation
        return matrix

    def quat(self):
        return Quaternion.from_rotation_matrix(self.linear)

    def __copy__(self):
        result = Affine3()
        result.linear = self.linear.copy()
        result.translation = self.translation.copy()
        return result

    def copy(self):
        return self.__copy__()

    def inv(self):
        return Affine3.from_matrix(np.linalg.inv(self.matrix()))

    def __mul__(self, other):
        return self.__rmul__(other)


    def __rmul__(self, other):
        return Affine3.from_matrix(np.matmul(self.matrix(), other.matrix()))

    def __str__(self):
        return self.matrix().__str__()

class Quaternion:
    def __init__(self, w=1, x=0, y=0, z=0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __copy__(self):
        return Quaternion(w=self.w, x=self.x, y=self.y, z=self.z)

    def copy(self):
        return self.__copy__()

    def as_vector(self, convention='wxyz'):
        if convention == 'wxyz':
            return np.array([self.w, self.x, self.y, self.z])
        elif convention == 'xyzw':
            return np.array([self.x, self.y, self.z, self.w])
        else:
            raise RuntimeError

    def normalize(self):
        q = self.as_vector()
        q = q / np.linalg.norm(q)
        self.w = q[0]
        self.x = q[1]
        self.y = q[2]
        self.z = q[3]

    def vec(self):
        return np.array([self.x, self.y, self.z])

    def error(self, quat_desired):
        return - self.w * quat_desired.vec() + quat_desired.w * self.vec() + np.matmul(skew_symmetric(quat_desired.vec()), self.vec())

    def rotation_matrix(self):
        """
        Transforms a quaternion to a rotation matrix.
        """
        n  = self.w
        ex = self.x
        ey = self.y
        ez = self.z

        R = np.eye(3)

        R[0, 0] = 2 * (n * n + ex * ex) - 1
        R[0, 1] = 2 * (ex * ey - n * ez)
        R[0, 2] = 2 * (ex * ez + n * ey)

        R[1, 0] = 2 * (ex * ey + n * ez)
        R[1, 1] = 2 * (n * n + ey * ey) - 1
        R[1, 2] = 2 * (ey * ez - n * ex)

        R[2, 0] = 2 * (ex * ez - n * ey)
        R[2, 1] = 2 * (ey * ez + n * ex)
        R[2, 2] = 2 * (n * n + ez * ez) - 1

        return R;

    @classmethod
    def from_vector(cls, vector, order='wxyz'):
        if order == 'wxyz':
            return cls(w=vector[0], x=vector[1], y=vector[2], z=vector[3])
        elif order == 'xyzw':
            return cls(w=vector[3], x=vector[0], y=vector[1], z=vector[2])
        else:
            raise ValueError('Order is not supported.')

    @classmethod
    def from_rotation_matrix(cls, R):
        """
        Transforms a rotation matrix to a quaternion.
        """

        q = [None] * 4

        tr = R[0, 0] + R[1, 1] + R[2, 2]

        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2  # S=4*qwh
            q[0] = 0.25 * S
            q[1] = (R[2, 1] - R[1, 2]) / S
            q[2] = (R[0, 2] - R[2, 0]) / S
            q[3] = (R[1, 0] - R[0, 1]) / S

        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
          S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
          q[0] = (R[2, 1] - R[1, 2]) / S
          q[1] = 0.25 * S
          q[2] = (R[0, 1] + R[1, 0]) / S
          q[3] = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
          S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
          q[0] = (R[0, 2] - R[2, 0]) / S
          q[1] = (R[0, 1] + R[1, 0]) / S
          q[2] = 0.25 * S
          q[3] = (R[1, 2] + R[2, 1]) / S
        else:
          S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
          q[0] = (R[1, 0] - R[0, 1]) / S
          q[1] = (R[0, 2] + R[2, 0]) / S
          q[2] = (R[1, 2] + R[2, 1]) / S
          q[3] = 0.25 * S

        result = q / lin.norm(q);
        return cls(w=result[0], x=result[1], y=result[2], z=result[3])

    @classmethod
    def from_dict(cls, quat):
        return cls(w=quat['w'], x=quat['x'], y=quat['y'], z=quat['z'])
    
    @classmethod
    def from_euler_angles(cls, roll, pitch, yaw):
        """
        Transforms euler angles to a quaternion
        """
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
            yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
            yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        return cls(x=qx, y=qy, z=qz, w=qw)

    def __str__(self):
        return str(self.w) + " + " + str(self.x) + "i +" + str(self.y) + "j + " + str(self.z) + "k"

    def rot_z(self, theta):
        mat = self.rotation_matrix()
        mat =  np.matmul(mat, rot_z(theta))
        new = Quaternion.from_rotation_matrix(mat)
        self.w = new.w
        self.x = new.x
        self.y = new.y
        self.z = new.z
        return self

    def log(self):
        if abs(self.w - 1) < 1e-12:
            return np.zero(3)
        vec_norm = lin.norm(self.vec())
        return math.atan2(vec_norm, self.w) * self.vec() / vec_norm;

    def mul(self, second):
        result = Quaternion()
        result.w = self.w * second.w - np.dot(self.vec(), second.vec())
        vec = self.w * second.vec() + second.w * self.vec() + np.cross(self.vec(), second.vec())
        result.x = vec[0]
        result.y = vec[1]
        result.z = vec[2]
        return result

    def inverse(self):
        result = Quaternion();
        temp = pow(self.w, 2) + pow(self.x, 2) + pow(self.y, 2) + pow(self.z, 2)
        result.w = self.w / temp
        result.x = - self.x / temp
        result.y = - self.y / temp
        result.z = - self.z / temp
        return result

    def log_error(self, desired):
        diff = self.mul(desired.inverse())
        if (diff.w < 0):
            diff.x = - diff.x
            diff.y = - diff.y
            diff.z = - diff.z
            diff.w = - diff.w
        return 2.0 * diff.log()

def rot_x(theta):
  rot = np.zeros((3, 3))
  rot[0, 0] = 1
  rot[0, 1] = 0
  rot[0, 2] = 0

  rot[1, 0] = 0
  rot[1, 1] = math.cos(theta)
  rot[1, 2] = - math.sin(theta)

  rot[2, 0] = 0
  rot[2, 1] = math.sin(theta)
  rot[2, 2] = math.cos(theta)

  return rot

def rot_y(theta):
  rot = np.zeros((3, 3))
  rot[0, 0] = math.cos(theta)
  rot[0, 1] = 0
  rot[0, 2] = math.sin(theta)

  rot[1, 0] = 0
  rot[1, 1] = 1
  rot[1, 2] = 0

  rot[2, 0] = - math.sin(theta)
  rot[2, 1] = 0
  rot[2, 2] = math.cos(theta)

  return rot

def rot_z(theta):
  rot = np.zeros((3, 3))
  rot[0, 0] = math.cos(theta)
  rot[0, 1] = - math.sin(theta)
  rot[0, 2] = 0

  rot[1, 0] = math.sin(theta)
  rot[1, 1] = math.cos(theta)
  rot[1, 2] = 0

  rot[2, 0] = 0
  rot[2, 1] = 0
  rot[2, 2] = 1

  return rot

def rotation_is_valid(R, eps=1e-8):
    # Columns should be unit
    for i in range(3):
        error = abs(np.linalg.norm(R[:, i]) - 1)
        if  error > eps:
            raise ValueError('Column ' + str(i) + ' of rotation matrix is not unit (error = ' + str(error) + ') precision: ' + str(eps) + ')')

    # Check that the columns are orthogonal
    if abs(np.dot(R[:, 0], R[:, 1])) > eps:
        raise ValueError('Column 0 and 1 of rotation matrix are not orthogonal (precision: ' + str(eps) + ')')
    if abs(np.dot(R[:, 0], R[:, 2])) > eps:
        raise ValueError('Column 0 and 2 of rotation matrix are not orthogonal (precision: ' + str(eps) + ')')
    if abs(np.dot(R[:, 2], R[:, 1])) > eps:
        raise ValueError('Column 2 and 1 of rotation matrix are not orthogonal (precision: ' + str(eps) + ')')

    # Rotation is right handed
    if not np.allclose(np.cross(R[:, 0], R[:, 1]), R[:, 2], rtol=0, atol=eps):
        raise ValueError('Rotation is not right handed (cross(x, y) != z for precision: ' + str(eps) + ')')
    if not np.allclose(np.cross(R[:, 2], R[:, 0]), R[:, 1], rtol=0, atol=eps):
        raise ValueError('Rotation is not right handed (cross(z, x) != y for precision: ' + str(eps) + ')')
    if not np.allclose(np.cross(R[:, 1], R[:, 2]), R[:, 0], rtol=0, atol=eps):
        raise ValueError('Rotation is not right handed (cross(y, z) != x for precision: ' + str(eps) + ')')

    return True

def transform_points(points, pos, quat, inv=False):
    '''Points are w.r.t. {A}. pos and quat is the frame {A} w.r.t {B}. Returns the list of points experssed w.r.t.
    {B}.'''
    assert points.shape[1] == 3
    matrix = np.eye(4)
    matrix[0:3, 3] = pos
    matrix[0:3, 0:3] = quat.rotation_matrix()
    if inv:
        matrix = np.linalg.inv(matrix)

    transformed_points = np.transpose(np.matmul(matrix, np.transpose(
        np.concatenate((points, np.ones((points.shape[0], 1))), axis=1))))[:, :3]
    return transformed_points

def transform_poses(poses, target_frame, target_inv=False):
    """
    Poses in Nx7 vectors with position & quaternion w.r.t. {A}.
    target_frame {B} 7x1 vector the target frame to express poses (w.r.t. again {A}
    """
    matrix = np.eye(4)
    matrix[0:3, 3] = target_frame[0:3]
    matrix[0:3, 0:3] = Quaternion.from_vector(target_frame[3:7]).rotation_matrix()
    transformed_poses = np.zeros((poses.shape[0], 7))
    matrix = np.linalg.inv(matrix)
    if target_inv:
        matrix = np.linalg.inv(matrix)

    for i in range(poses.shape[0]):
        matrix2 = np.eye(4)
        matrix2[0:3, 3] = poses[i, 0:3]
        matrix2[0:3, 0:3] = Quaternion.from_vector(poses[i, 3:7]).rotation_matrix()
        transformed = np.matmul(matrix, matrix2)
        transformed_poses[i, 0:3] = transformed[0:3, 3]
        transformed_poses[i, 3:7] = Quaternion.from_rotation_matrix(transformed[0:3, 0:3]).as_vector()
    return transformed_poses

