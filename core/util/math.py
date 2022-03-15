import scipy
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import math


def triangle_area(t):
    """Calculates the area of a triangle defined given its 3 vertices. n_vertices x n_dims =  3 x 2"""
    return (1 / 2) * abs((t[0][0] - t[2][0]) * (t[1][1] - t[0][1]) - (t[0][0] - t[1][0]) * (t[2][1] - t[0][1]))


def transform_points(points, pos, quat):
    """
    Points are w.r.t. {A}. pos and quat is the frame {A} w.r.t {B}. Returns the list of points experssed w.r.t. {B}.
    """
    assert points.shape[1] == 3
    matrix = np.eye(4)
    matrix[0:3, 3] = pos
    matrix[0:3, 0:3] = quat.rotation_matrix()

    transformed_points = np.transpose(np.matmul(matrix, np.transpose(
        np.concatenate((points, np.ones((points.shape[0], 1))), axis=1))))[:, :3]
    return transformed_points


class LineSegment2D:
    def __init__(self, p1, p2):
        self.p1 = p1.copy()
        self.p2 = p2.copy()

    def get_point(self, lambd):
        assert lambd >=0 and lambd <=1
        return (1 - lambd) * self.p1 + lambd * self.p2

    def get_lambda(self, p3):
        lambd = (p3[0] - self.p1[0]) / (self.p2[0] - self.p1[0])
        lambd_2 = (p3[1] - self.p1[1]) / (self.p2[1] - self.p1[1])
        if abs(lambd - lambd_2) > 1e-5:
            return None
        return lambd

    def get_intersection_point(self, line_segment, belong_self=True, belong_second=True):
        '''See wikipedia https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line'''
        x1 = self.p1[0]
        y1 = self.p1[1]
        x2 = self.p2[0]
        y2 = self.p2[1]

        x3 = line_segment.p1[0]
        y3 = line_segment.p1[1]
        x4 = line_segment.p2[0]
        y4 = line_segment.p2[1]

        if abs((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)) < 1e-10:
            return None

        t =  ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / \
             ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        u = - ((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / \
              ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

        p = np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])

        if belong_self and belong_second:
            if t >=0 and t <= 1 and u >= 0 and u <= 1:
                return p
            else:
                return None
        elif (belong_self and t >=0 and t <= 1) or (belong_second and u >= 0 and u <= 1):
            return p
        elif not belong_self and not belong_second:
            return p
        return None

    def get_first_intersection_point(self, line_segments):
        for line_segment in line_segments:
            result = self.get_intersection_point(line_segment)
            if result is not None:
                break
        return result

    def norm(self):
        return np.linalg.norm(self.p1 - self.p2)

    def __str__(self):
        return self.p1.__str__() + ' ' + self.p2.__str__()

    def array(self):
        result = np.zeros((2, 2))
        result[0, :] = self.p1
        result[1, :] = self.p2
        return result

    @staticmethod
    def plot_line_segments(line_segments, points=[]):
        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(line_segments) + len(points))))
        lines = []
        i = 0
        for line_segment in line_segments:
            c = next(color)
            plt.plot(line_segment.p1[0], line_segment.p1[1], color=c, marker='o')
            plt.plot(line_segment.p2[0], line_segment.p2[1], color=c, marker='.')
            plt.plot([line_segment.p1[0], line_segment.p2[0]], [line_segment.p1[1], line_segment.p2[1]], color=c, linestyle='-')
            lines.append(Line2D([0], [0], label='LineSegment_' + str(i), color=c))
            i += 1

        i = 0
        for point in points:
            c = next(color)
            plt.plot(point[0], point[1], color=c, marker='o')
            lines.append(Line2D([0], [0], marker='o', label='Point_' + str(i), color=c))
            i += 1

        plt.legend(handles=lines)
        plt.show()

    def rotate(self, theta):
        '''Theta in rad'''
        rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        self.p1 = np.matmul(rot, self.p1)
        self.p2 = np.matmul(rot, self.p2)
        return self

    def translate(self, p):
        self.p1 += p
        self.p2 += p
        return self

    def plot(self, color=(0, 0, 0, 1), ax=None):
        if ax is None:
            fig = plt.figure()
            ax = Axes3D(fig)
        ax.plot([self.p1[0], self.p2[0]],
                [self.p1[1], self.p2[1]],
                [self.p1[2], self.p2[2]], color=color, linestyle='-')
        return ax


class ConvexHull(scipy.spatial.ConvexHull):
    """
    Extendes scipy's ConvexHull to compute the centroid and to represent convex hull in the form of line segments.
    More information for the parent class here:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
    """
    def centroid(self):
        """
        Calculates the centroid of a 2D convex hull

        Returns
        -------
        np.array :
            The 2D centroid
        """
        hull_points = self.points[self.vertices]
        tri = Delaunay(hull_points)
        triangles = np.zeros((tri.simplices.shape[0], 3, 2))
        for i in range(len(tri.simplices)):
            for j in range(3):
                triangles[i, j, 0] = hull_points[tri.simplices[i, j], 0]
                triangles[i, j, 1] = hull_points[tri.simplices[i, j], 1]

        centroids = np.mean(triangles, axis=1)

        triangle_areas = np.zeros(len(triangles))
        for i in range(len(triangles)):
            triangle_areas[i] = triangle_area(triangles[i, :, :])

        weights = triangle_areas / np.sum(triangle_areas)

        centroid = np.average(centroids, axis=0, weights=weights)

        return centroid

    def line_segments(self):
        """
        Returns the convex hull as a list of line segments (LineSegment2D).

        Returns
        -------

        list :
            The list of line segments
        """
        hull_points = np.zeros((len(self.vertices), 2))
        segments = []
        hull_points[0, 0] = self.points[self.vertices[0], 0]
        hull_points[0, 1] = self.points[self.vertices[0], 1]
        i = 1
        for i in range(1, len(self.vertices)):
            hull_points[i, 0] = self.points[self.vertices[i], 0]
            hull_points[i, 1] = self.points[self.vertices[i], 1]
            segments.append(LineSegment2D(hull_points[i - 1, :], hull_points[i, :]))
        segments.append(LineSegment2D(hull_points[i, :], hull_points[0, :]))
        return segments

    def plot(self, ax=None):
        """
        Plots the convex hull in 2D.

        ax : matplotlib.axes
            An axes object to use for plotting in an existing plot
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.points[:, 0], self.points[:, 1], 'o')
        centroid = self.centroid()
        ax.plot(centroid[0], centroid[1], 'o')
        for simplex in self.simplices:
            ax.plot(self.points[simplex, 0], self.points[simplex, 1], 'k')
        return ax