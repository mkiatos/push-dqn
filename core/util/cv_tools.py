import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import cv2

'''
Computer Vision Utils
============
'''


class PinholeCameraIntrinsics:
    """
    PinholeCameraIntrinsics class stores intrinsic camera matrix,
    and image height and width.
    """
    def __init__(self, width, height, fx, fy, cx, cy):

        self.width, self.height = width, height
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy

    @classmethod
    def from_params(cls, params):
        width, height = params['width'], params['height']
        fx, fy = params['fx'], params['fy']
        cx, cy = params['cx'], params['cy']
        return cls(width, height, fx, fy, cx, cy)

    def get_intrinsic_matrix(self):
        camera_matrix = np.array(((self.fx, 0, self.cx),
                                  (0, self.fy, self.cy),
                                  (0, 0, 1)))
        return camera_matrix

    def get_focal_length(self):
        return self.fx, self.fy

    def get_principal_point(self):
        return self.cx, self.cy

    def back_project(self, p, z):
        x = (p[0] - self.cx) * z / self.fx
        y = (p[1] - self.cy) * z / self.fy
        return np.array([x, y, z])


class PointCloud(o3d.geometry.PointCloud):
    def __init__(self):
        super(PointCloud, self).__init__()

    @staticmethod
    def from_depth(depth, camera_intrinsics):
        """
        Creates a point cloud from a depth image given the camera
        intrinsics parameters.

        Parameters
        ----------
        depth: np.array
            The input image.
        camera_intrinsics: PinholeCameraIntrinsics object
            Intrinsics parameters of the camera.

        Returns
        -------
        o3d.geometry.PointCloud
            The point cloud of the scene.
        """
        depth = depth
        width, height = depth.shape
        c, r = np.meshgrid(np.arange(height), np.arange(width), sparse=True)
        valid = (depth > 0)
        z = np.where(valid, depth, 0)
        x = np.where(valid, z * (c - camera_intrinsics.cx) / camera_intrinsics.fx, 0)
        y = np.where(valid, z * (r - camera_intrinsics.cy) / camera_intrinsics.fy, 0)
        pcd = np.dstack((x, y, z))
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.reshape(-1, 3)))


class Feature:
    """
    heightmap: 2d array representing the topography of the scene
    """
    def __init__(self, heightmap):
        self.heightmap = heightmap.copy()
        self.h, self.w = heightmap.shape
        self.center = [int(self.w / 2),
                       int(self.h / 2)]

    def get_heightmap(self, depth):
        max_depth = np.max(depth)
        depth[depth == 0] = max_depth
        return max_depth - depth

    def increase_canvas_size(self, x, y):
        assert x >= self.h and y >= self.w
        new_center = [int(x / 2), int(y / 2)]
        if len(self.heightmap.shape) == 3:
            new_shape = (x, y, self.heightmap.shape[2])
        else:
            new_shape = (x, y)
        new_canvas = np.zeros(new_shape, dtype=self.heightmap.dtype)
        first_row = new_center[0] - self.center[0]
        first_col = new_center[1] - self.center[1]
        last_row = first_row + self.h
        last_col = first_col + self.w
        new_canvas[first_row:last_row, first_col:last_col] = self.heightmap.copy()
        return Feature(new_canvas)

    def crop(self, x, y):
        """
        Crop the height map around the center with the given size
        """
        cropped_heightmap = self.heightmap[self.center[1] - y:self.center[1] + y,
                                           self.center[0] - x:self.center[0] + x]
        return Feature(cropped_heightmap)

    def mask_in(self, mask):
        """
        Remove from height map the pixels that do not belong to the mask
        """
        mask = mask.astype(np.int8)
        mask_in_heightmap = cv2.bitwise_and(self.heightmap, self.heightmap, mask=mask)
        return Feature(mask_in_heightmap)

    def mask_out(self, mask):
        """
        Remove from height map the pixels that belong to the mask
        """
        mask = (255 - mask).astype(np.int8)
        mask_out_heightmap = cv2.bitwise_and(self.heightmap, self.heightmap, mask=mask)
        return Feature(mask_out_heightmap)

    def pooling(self, kernel=[4, 4], stride=4, mode='AVG'):
        """
        Pooling operations on the depth and mask
        """
        out_width = int((self.w - kernel[0]) / stride + 1)
        out_height = int((self.h - kernel[1]) / stride + 1)

        # Perform max/avg pooling based on the mode
        pool_heightmap = np.zeros((out_width, out_height))
        for x in range(out_width):
            for y in range(out_height):
                corner = (x * stride + kernel[0], y * stride + kernel[1])
                region = [(corner[0] - kernel[0], corner[1] - kernel[1]), corner]
                if mode == 'AVG':
                    pool_heightmap[y, x] = np.sum(self.heightmap[region[0][1]:region[1][1],
                                                                 region[0][0]:region[1][0]]) / (stride * stride)
                elif mode == 'MAX':
                    pool_heightmap[y, x] = np.max(self.heightmap[region[0][1]:region[1][1],
                                                                 region[0][0]:region[1][0]])
        return Feature(pool_heightmap)

    def translate(self, tx, ty, inverse=False):
        """
        Translates the heightmap
        """
        t_x = self.center[0] - int(tx)
        t_y = self.center[1] - int(ty)
        if inverse:
            t_x *= -1
            t_y *= -1
        t = np.float32([[1, 0, t_x], [0, 1, t_y]])
        translated_heightmap = cv2.warpAffine(self.heightmap, t, (self.w, self.h))
        return Feature(translated_heightmap)

    def rotate(self, theta):
        """
        Rotate the heightmap around its center
        theta: Rotation angle in degrees. Positive values mean counter-clockwise rotation .
        """
        scale = 1.0
        rot = cv2.getRotationMatrix2D((self.center[0], self.center[1]),
                                      theta, scale)
        rotated_heightmap = cv2.warpAffine(self.heightmap, rot, (self.w, self.h))
        return Feature(rotated_heightmap)

    def non_zero_pixels(self):
        return np.argwhere(self.heightmap > 0).shape[0]

    def array(self):
        """
        Return 2d array
        """
        return self.heightmap

    def flatten(self):
        """
        Flatten to 1 dim and return to use as 1dim vector
        """
        return self.heightmap.flatten()

    def plot(self, name='height_map', grayscale=True):
        """
        Plot the heightmap
        """
        if grayscale:
            plt.imshow(self.heightmap, cmap='gray',
                       vmin=np.min(self.heightmap),
                       vmax=np.max(self.heightmap))
        else:
            plt.imshow(self.heightmap)
        plt.title(name)
        plt.show()
        return self

    def normalize(self, max_height):
        normalized = self.heightmap / max_height
        normalized[normalized > 1] = 1
        normalized[normalized < 0] = 0
        return Feature(normalized)

    def resize(self, dim):
        resized_img = cv2.resize(self.heightmap, dim, interpolation=cv2.INTER_AREA)
        return Feature(resized_img)


def calc_m_per_pxl(camera_intrinsics, depth, surface_mask, random_points=10):
    """
    Calculates how many meters correspond to one pixel when moving on a specific plane

    Parameters
    ----------
    camera_intrinsics : PinholeCameraIntrinsics
        The intrinsics of the camera
    depth : np.ndarray
        A depth image from the scene
    surface_mask : np.ndarray
        The mask of the plane (0 in non-plane pixels, 255 in plane's pixels)
    random_points : int
        The number of random samples for calculating the distances
    """
    np.random.seed(0)
    indices = np.argwhere(surface_mask == 255)
    samples = np.random.randint(0, indices.shape[0], random_points * 2)

    data = []
    for i in range(random_points):
        # Calculate the distance between two points in image
        p_a = np.array([indices[samples[i], 1], indices[samples[i], 0]])
        p_b = np.array([indices[samples[random_points + i], 1], indices[samples[random_points + i], 0]])
        dist_in_pxl = np.linalg.norm(p_b - p_a)

        # Calculate the distance of the same points backprojected on 3D
        p_a_3d = camera_intrinsics.back_project(p_a, depth[p_a[1], p_a[0]])
        p_b_3d = camera_intrinsics.back_project(p_b, depth[p_b[1], p_b[0]])
        dist_in_m = np.linalg.norm(p_b_3d[:2] - p_a_3d[:2])

        data.append(dist_in_m/dist_in_pxl)

    return np.mean(data)
