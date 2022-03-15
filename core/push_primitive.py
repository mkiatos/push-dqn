from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import copy
import math


from util.utils import min_max_scale
from util.orientation import Quaternion, Affine3
from util.math import LineSegment2D, ConvexHull, transform_points


class Push:
    """
    A pushing action of two 3D points for init and final pos. Every pushing
    action should inherite this class.
    """

    def __init__(self, p1=None, p2=None):
        self.p1 = copy.copy(p1)
        self.p2 = copy.copy(p2)

    def __call__(self, p1, p2):
        self.p1 = p1.copy()
        self.p2 = p2.copy()

    def __str__(self):
        return self.p1.__str__() + ' ' + self.p2.__str__()

    def get_init_pos(self):
        return self.p1

    def get_final_pos(self):
        return self.p2

    def get_duration(self, distance_per_sec=0.1):
        return np.linalg.norm(self.get_init_pos() - self.get_final_pos()) / distance_per_sec

    def translate(self, p):
        self.p1 += p
        self.p2 += p
        return self

    def rotate(self, quat):
        """
        Rot: rotation matrix
        """
        self.p1 = np.matmul(quat.rotation_matrix(), self.p1)
        self.p2 = np.matmul(quat.rotation_matrix(), self.p2)
        return self

    def transform(self, pos, quat):
        """
        The ref frame
        """
        assert isinstance(pos, np.ndarray) and pos.shape == (3,)
        assert isinstance(quat, Quaternion)
        tran = Affine3.from_vec_quat(pos, quat)
        self.p1 = np.matmul(tran.matrix(), np.append(self.p1, 1))[:3]
        self.p2 = np.matmul(tran.matrix(), np.append(self.p2, 1))[:3]

    def plot(self, ax=None, show=False):
        if ax is None:
            fig = plt.figure()
            ax = Axes3D(fig)
        color = [0, 0, 0]
        ax.plot(self.p1[0], self.p1[1], self.p1[2], color=color, marker='o')
        ax.plot(self.p2[0], self.p2[1], self.p2[2], color=color, marker='>')
        # ax.plot(self.p2[0], self.p2[1], color=color, marker='.')
        # ax.plot([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]], color=color, linestyle='-')

        ax.plot([self.p1[0], self.p2[0]],
                [self.p1[1], self.p2[1]],
                [self.p1[2], self.p2[2]],
                color=color, linestyle='-')

        if show:
            plt.show()
        return ax


class PushTarget(Push):
    def __init__(self, push_distance_range=None, init_distance_range=None):
        self.push_distance_range = push_distance_range
        self.init_distance_range = init_distance_range
        super(PushTarget, self).__init__()

    def __call__(self, theta, push_distance, distance, normalized=False, push_distance_from_target=True):
        theta_ = theta
        push_distance_ = push_distance
        distance_ = distance
        if normalized:
            theta_ = min_max_scale(theta, range=[-1, 1], target_range=[-np.pi, np.pi])
            distance_ = min_max_scale(distance, range=[-1, 1], target_range=self.init_distance_range)
            push_distance_ = min_max_scale(push_distance, range=[-1, 1], target_range=self.push_distance_range)
        assert push_distance_ >= 0
        assert distance_ >= 0
        p1 = np.array([-distance_ * np.cos(theta_), -distance_ * np.sin(theta_), 0])
        p2 = np.array([push_distance_ * math.cos(theta_), push_distance_ * math.sin(theta_), 0])
        if not push_distance_from_target:
            p2 += p1
        super(PushTarget, self).__call__(p1, p2)


class PushObstacle(Push):
    """
    The push-obstacle primitive.

    Parameters
    ----------
    push_distance_range : list
        Min and max value for the pushing distance in meters.
    offset : float
        Offset in meters which will be added on the height above the push. Use it to ensure that the finger will
        not slide on the target
    """
    def __init__(self, push_distance_range=None, offset=0.003):
        self.push_distance_range = push_distance_range
        self.offset = offset
        super(PushObstacle, self).__init__()

    def __call__(self, theta, push_distance, target_size_z, normalized=False):
        theta_ = theta
        push_distance_ = push_distance
        if normalized:
            assert self.push_distance_range is not None, "push_distance_range cannot be None for normalized inputs."
            theta_ = min_max_scale(theta, range=[-1, 1], target_range=[-np.pi, np.pi])
            push_distance_ = min_max_scale(push_distance, range=[-1, 1], target_range=self.push_distance_range)

        p1 = np.array([0, 0, target_size_z + self.offset])
        p2 = np.array([push_distance_ * math.cos(theta_), push_distance_ * math.sin(theta_),
                       target_size_z + self.offset])
        super(PushObstacle, self).__call__(p1, p2)


class PushAndAvoidTarget(PushTarget):
    """
    A 2D push for pushing target which uses the 2D convex hull of the object to enforce obstacle avoidance.
    convex_hull: A list of Linesegments2D. Should by in order cyclic, in order to calculate the centroid correctly
    Theta, push_distance, distance assumed to be in [-1, 1]
    """

    def __init__(self, finger_size, push_distance_range=None, init_distance_range=None):
        self.finger_size = finger_size
        super(PushAndAvoidTarget, self).__init__(push_distance_range=push_distance_range,
                                                 init_distance_range=init_distance_range)

    def __call__(self, theta, push_distance, distance, convex_hull, normalized=False, push_distance_from_target=True):
        theta_ = theta
        push_distance_ = push_distance
        distance_ = distance
        if normalized:
            theta_ = min_max_scale(theta, range=[-1, 1], target_range=[-np.pi, np.pi])
            distance_ = min_max_scale(distance, range=[-1, 1], target_range=self.init_distance_range)
            push_distance_ = min_max_scale(push_distance, range=[-1, 1], target_range=self.push_distance_range)
        assert push_distance_ >= 0
        assert distance_ >= 0

        # Calculate offset
        assert isinstance(convex_hull, ConvexHull)

        # Calculate the initial point p1 from convex hull
        # -----------------------------------------------
        # Calculate the intersection point between the direction of the
        # push theta and the convex hull (four line segments)
        direction = np.array([math.cos(theta_), math.sin(theta_)])

        # Quat of finger
        y_direction = np.array([direction[0], direction[1], 0])
        x = np.cross(y_direction, np.array([0, 0, -1]))

        rot_mat = np.array([[x[0], y_direction[0], 0],
                            [x[1], y_direction[1], 0],
                            [x[2], y_direction[2], -1]])

        quat = Quaternion.from_rotation_matrix(rot_mat)

        bbox_corners_object = np.array([[self.finger_size[0], self.finger_size[1], 0],
                                        [self.finger_size[0], -self.finger_size[1], 0],
                                        [-self.finger_size[0], -self.finger_size[1], 0],
                                        [-self.finger_size[0], self.finger_size[1], 0]])

        line_segment = LineSegment2D(np.array([0, 0]), 10 * direction)
        init_point = line_segment.get_first_intersection_point(convex_hull.line_segments())
        found = False
        step = 0.002
        offset = 0
        point = init_point
        point = np.array([point[0], point[1], 0])
        while not found:
            points = transform_points(bbox_corners_object, point, quat)
            offset += step
            point = init_point + offset * direction
            point = np.array([point[0], point[1], 0])
            cross_sec = [False, False, False, False]
            linesegs = [LineSegment2D(points[0], points[1]),
                        LineSegment2D(points[1], points[2]),
                        LineSegment2D(points[2], points[3]),
                        LineSegment2D(points[3], points[0])]

            # ax = None
            # for i in range(len(linesegs)):
            #     ax = linesegs[i].plot(ax=ax)
            #     print(ax)
            # convex_hull.plot(ax=ax)
            # plt.show()

            for i in range(len(linesegs)):
                if linesegs[i].get_first_intersection_point(convex_hull.line_segments()) is None:
                    cross_sec[i] = True
            if np.array(cross_sec).all():
                found = True

        offset += np.linalg.norm(init_point)
        super(PushAndAvoidTarget, self).__call__(theta_,
                                                 push_distance_,
                                                 distance_ + offset,
                                                 normalized=False,
                                                 push_distance_from_target=push_distance_from_target)
