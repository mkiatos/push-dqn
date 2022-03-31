import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import cv2

from push_primitive import PushAndAvoidTarget, PushObstacle
from clt_core.util.cv_tools import PinholeCameraIntrinsics, PointCloud, Feature
from clt_core.util.pybullet import get_camera_pose
from clt_core.core import MDP
from clt_core.util.math import min_max_scale

CROP_TABLE = 193


def get_heightmap(obs):
    """
    Computes the heightmap based on the 'depth' and 'seg'. In this heightmap table pixels has value zero,
    objects > 0 and everything below the table <0.

    Parameters
    ----------
    obs : dict
        The dictionary with the visual and full state of the environment.

    Returns
    -------
    np.ndarray :
        Array with the heightmap.
    """
    rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']
    objects = obs['full_state']['objects']

    # Compute heightmap
    table_id = next(x.body_id for x in objects if x.name == 'table')
    depthcopy = depth.copy()
    table_depth = np.max(depth[seg == table_id])
    depthcopy[seg == table_id] = table_depth
    heightmap = table_depth - depthcopy
    return heightmap


def empty_push(obs, next_obs, eps=0.005):
    """
    Checks if the objects have been moved
    """

    for prev_obj in obs['full_state']['objects']:
        if prev_obj.name in ['table', 'plane']:
            continue

        for obj in next_obs['full_state']['objects']:
            if prev_obj.body_id == obj.body_id:
                if np.linalg.norm(prev_obj.pos - obj.pos) > eps:
                    return False
    return True


def get_distances_from_target(obs):
    objects = obs['full_state']['objects']

    # Get target pose from full state
    target = next(x for x in objects if x.name == 'target')
    target_pose = np.eye(4)
    target_pose[0:3, 0:3] = target.quat.rotation_matrix()
    target_pose[0:3, 3] = target.pos

    # Compute the distances of the obstacles from the target
    distances_from_target = []
    for obj in objects:
        if obj.name in ['target', 'table', 'plane']:
            continue

        # Transform the objects w.r.t. target (reduce variability)
        obj_pose = np.eye(4)
        obj_pose[0:3, 0:3] = obj.quat.rotation_matrix()
        obj_pose[0:3, 3] = obj.pos

        # distance = get_distance_of_two_bbox(target_pose, target.size, obj_pose, obj.size)
        points = p.getClosestPoints(target.body_id, obj.body_id, distance=10)
        distance = np.linalg.norm(np.array(points[0][5]) - np.array(points[0][6]))

        # points = p.getClosestPoints(target.body_id, obj.body_id, distance=10)
        # distance = np.linalg.norm(np.array(points[0][5]) - np.array(points[0][6]))

        distances_from_target.append(distance)
    return np.array(distances_from_target)


class DiscreteMDP(MDP):
    def __init__(self, params):
        super(DiscreteMDP, self).__init__(name='discre_mdp', params=params)
        # TODO: hard-coded values
        self.pinhole_camera_intrinsics = PinholeCameraIntrinsics.from_params(params['env']['camera']['intrinsics'])
        camera_pos, camera_quat = get_camera_pose(np.array(params['env']['camera']['pos']),
                                                  np.array(params['env']['camera']['target_pos']),
                                                  np.array(params['env']['camera']['up_vector']))
        self.camera_pose = np.eye(4)
        self.camera_pose[0:3, 0:3] = camera_quat.rotation_matrix()
        self.camera_pose[0:3, 3] = camera_pos

        # params from yaml
        self.m_per_pixel = 400
        self.singulation_distance = 0.03

        self.surface_size = params['env']['workspace']['size']

        self.singulation_distance = params['mdp']['singulation_distance']
        self.nr_discrete_actions = params['mdp']['nr_discrete_actions']
        self.nr_primitives = params['mdp']['nr_primitives']
        self.push_distance = params['mdp']['push_distance']

    def get_scene_points(self, obs):
        """
        Returns the scene point cloud transformed w.r.t. target. We remove the
        points that belong to table
        """
        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']
        objects = obs['full_state']['objects']

        # Get target pose from full state
        target_obj = next(x for x in objects if x.name == 'target')
        target_pose = np.eye(4)
        target_pose[0:3, 0:3] = target_obj.quat.rotation_matrix()
        target_pose[0:3, 3] = target_obj.pos

        # Create scene point cloud
        point_cloud = PointCloud.from_depth(depth, self.pinhole_camera_intrinsics)

        # Transform point cloud w.r.t. target frame
        point_cloud.transform(np.matmul(np.linalg.inv(target_pose), self.camera_pose))

        # Keep points only above the table
        z = np.asarray(point_cloud.points)[:, 2]
        ids = np.where(z > 0.0)
        above_pts = point_cloud.select_by_index(ids[0].tolist())

        # import open3d as o3d
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # o3d.visualization.draw_geometries([above_pts, frame])

        return np.asarray(above_pts.points)

    @staticmethod
    def get_heightmap(point_cloud, shape=(100, 100), grid_step=0.005):
        """
        Computes the heightmap of the scene given the aligned point cloud.
        """
        width = shape[0]
        height = shape[1]

        height_grid = np.zeros((height, width), dtype=np.float32)

        for i in range(point_cloud.shape[0]):
            x = point_cloud[i][0]
            y = point_cloud[i][1]
            z = point_cloud[i][2]

            idx_x = int(np.floor((x / grid_step) + 0.5)) + int(width / 2)
            idx_y = int(np.floor((y / grid_step) + 0.5)) + int(height / 2)

            if 0 < idx_x < width - 1 and 0 < idx_y < height - 1:
                if height_grid[idx_y][idx_x] < z:
                    height_grid[idx_y][idx_x] = z

        return height_grid

    def extract_features(self, heightmap, mask, plot=False):
        h, w = heightmap.shape
        cx = int(w / 2)
        cy = int(h / 2)

        target_ids = np.argwhere(mask > 0)
        x_min = np.min(target_ids[:, 1])
        x_max = np.max(target_ids[:, 1])
        y_min = np.min(target_ids[:, 0])
        y_max = np.max(target_ids[:, 0])
        bbox = np.array([int((x_max - x_min)/2.0), int((y_max - y_min)/2.0)])
        cx1 = cx - int(bbox[0]+0.5)
        cy1 = cy - int(bbox[1]+0.5)

        features = []

        # Define cells for computing the target features
        target_up_left_corners = [(cx1, cy1), (cx, cy1), (cx1, cy), (cx, cy)]

        # Compute features for the target
        for corner in target_up_left_corners:
            cropped = heightmap[corner[1]:corner[1] + 2*bbox[1],
                                corner[0]:corner[0] + 2*bbox[0]]
            features.append(np.mean(cropped))

        # Define the up left corners for each 32x32 region around the target [f_up, f_right, f_down, f_left]
        feature_area = 16
        kernel = [4, 4]
        stride = 4
        up_left_corners = [(int(cx - feature_area), int(cy - bbox[1] - 2 * feature_area)),
                           (int(cx + bbox[0]), int(cy - feature_area)),
                           (int(cx - feature_area), int(cy + bbox[1])),
                           (int(cx - bbox[0] - 2 * feature_area), int(cy - feature_area))]

        # Compute f_up, f_right, f_down, f_left
        for up_left_corner in up_left_corners:
            center = [up_left_corner[0] + feature_area, up_left_corner[1] + feature_area]
            cropped = heightmap[center[1] - feature_area:center[1] + feature_area,
                                center[0] - feature_area:center[0] + feature_area]
            features += Feature(cropped).pooling(kernel, stride).flatten().tolist()

        if plot:
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1)
            ax.imshow(heightmap)
            rect = patches.Rectangle((cx1, cy1), int(2*bbox[0]), int(2*bbox[1]),
                                     linewidth=1, edgecolor='r', facecolor='none')
            # plt.plot(cx, cy, 'ro')
            ax.add_patch(rect)
            for pt in up_left_corners:
                rect = patches.Rectangle(pt, feature_area * 2, feature_area * 2,
                                         linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            plt.show()

        # Normalize features to 0-1
        # print(np.min(np.asarray(features)), np.max(np.asarray(features)))
        features = min_max_scale(np.asarray(features),
                                 range=[0, np.max(np.asarray(features))],
                                 target_range=[0, 1])
        # print(np.min(features), np.max(features))

        return np.asarray(features)

    def state_representation(self, obs):
        # Get target pose from full state
        target = next(x for x in obs['full_state']['objects'] if x.name == 'target')

        # Get scene point cloud
        # points = self.get_scene_points(obs)

        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']
        target_id = next(x.body_id for x in obs['full_state']['objects'] if x.name == 'target')

        heightmap = get_heightmap(obs)
        heightmap[heightmap < 0] = 0  # Set negatives (everything below table) the same as the table in order to
        # properly translate it
        # print(len(np.argwhere(seg == target_id)))
        if len(np.argwhere(seg == target_id)) == 0:
            return np.zeros(262,)
        target_centroid = np.mean(np.argwhere(seg == target_id), axis=0)

        # Get target pose from full state
        target_obj = next(x for x in obs['full_state']['objects'] if x.name == 'target')
        target_pose = np.eye(4)
        target_pose[0:3, 0:3] = target_obj.quat.rotation_matrix()
        target_pose[0:3, 3] = target_obj.pos
        angle = np.arccos(np.dot(np.array([1, 0, 0]), target_pose[0:3, 0].transpose()))

        heightmap = Feature(heightmap).translate(tx=target_centroid[1], ty=target_centroid[0]).\
                                       rotate(-angle * 180 / np.pi).\
                                       crop(CROP_TABLE, CROP_TABLE).array()
        heightmap = cv2.resize(heightmap, (100, 100), interpolation=cv2.INTER_NEAREST)

        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[seg == target_id] = 255
        mask = Feature(mask).translate(tx=target_centroid[1], ty=target_centroid[0]).\
                                       rotate(-angle * 180 / np.pi).\
                                       crop(CROP_TABLE, CROP_TABLE).array()
        mask = cv2.resize(mask, (100, 100), interpolation=cv2.INTER_NEAREST)

        # # Generate heightmap
        # heightmap = self.get_heightmap(points)

        # Extract features
        features = self.extract_features(heightmap=heightmap, mask=mask)
        target_pos = np.array([target.pos[0] / (self.surface_size[0] / 2.0),
                               target.pos[1] / (self.surface_size[1] / 2.0)])
        features = np.concatenate((features, target_pos))

        return features

    def point_around_target(self, obs, bbox_limit=0.01):
        """
        Extracts the points that lie inside the singulation area
        """
        target = next(x for x in obs['full_state']['objects'] if x.name == 'target')
        points = self.get_scene_points(obs)

        points_around = []
        for p in points:
            if (-target.size[0] - bbox_limit > p[0] > -target.size[0] - self.singulation_distance - bbox_limit or
                target.size[0] + bbox_limit < p[0] < target.size[0] + self.singulation_distance + bbox_limit) and \
                    -target.size[1] < p[1] < target.size[1]:
                points_around.append(p)
            if (-target.size[1] - bbox_limit > p[1] > -target.size[1] - self.singulation_distance - bbox_limit or
                target.size[1] + bbox_limit < p[1] < target.size[1] + self.singulation_distance + bbox_limit) and \
                    -target.size[0] < p[0] < target.size[0]:
                points_around.append(p)

        return points_around

    def fallen(self, obs, next_obs):
        target = next(x for x in next_obs['full_state']['objects'] if x.name == 'target')
        if target.pos[2] < 0:
            return True
        return False

    def reward(self, obs, next_obs, action):

        # collision or empty push or fallen
        if next_obs['collision'] or empty_push(obs, next_obs) or self.fallen(obs, next_obs):
            return -10.0

        # singulation
        # if len(self.point_around_target(next_obs)) == 0:
        if all(dist > self.singulation_distance for dist in get_distances_from_target(next_obs)):
            return 10.0

        # Push near the bin walls
        target = next(x for x in next_obs['full_state']['objects'] if x.name == 'target')
        distances = np.asarray([self.surface_size[0] / 2.0 - target.pos[0], self.surface_size[0] / 2.0 + target.pos[0],
                                self.surface_size[1] / 2.0 - target.pos[1], self.surface_size[1] / 2.0 + target.pos[1]])
        distances -= 0.05
        if (distances < 0).any():
            return -5.0

        return -1.0

    def terminal(self, obs, next_obs):
        """
        Parameters
        ----------
        obs
        next_obs

        Returns
        -------
        terminal : int
            0: not a terminal state
            2: singulation
            3: fallen-off the table
        """
        # In case the target is singulated or falls of the table the episode is singulated

        if next_obs['collision']:
            return -1

        if empty_push(obs, next_obs):
            return -2

        target = next(x for x in next_obs['full_state']['objects'] if x.name == 'target')
        if target.pos[2] < 0:
            return 3
        # elif len(self.point_around_target(next_obs)) == 0:
        #     return 2
        if all(dist > self.singulation_distance for dist in get_distances_from_target(next_obs)):
            return 2
        else:
            return 0

    def action(self, obs, action):
        target = next(x for x in obs['full_state']['objects'] if x.name == 'target')

        # Compute the discrete angle theta
        theta = action * 2 * np.pi / (self.nr_discrete_actions / self.nr_primitives)

        # Choose the action
        if action > int(self.nr_discrete_actions / self.nr_primitives) - 1:
            # Push obstacle: 4-7
            push = PushObstacle()
            push(theta=theta, push_distance=self.push_distance, target_size_z=target.size[2])
        else:
            # Push target: 0-3
            push = PushAndAvoidTarget(finger_size=obs['full_state']['finger'])
            push(theta=theta, push_distance=self.push_distance, distance=0.0,
                 convex_hull=target.convex_hull(oriented=False))

        push.transform(target.pos, target.quat)

        return push.p1.copy(), push.p2.copy()


    def init_state_is_valid(self, obs):
        rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']
        target = next(x for x in obs['full_state']['objects'] if x.name == 'target')

        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        mask[seg == target.body_id] = 255
        mask = Feature(mask).crop(CROP_TABLE, CROP_TABLE).array()

        if (mask == 0).all() or target.pos[2] < 0:
            return False

        return True
