import numpy as np
import pybullet as p

from push_primitive import PushAndAvoidTarget, PushObstacle
from util.cv_tools import PinholeCameraIntrinsics, PointCloud, Feature
from util.pybullet import get_camera_pose


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


class MDP:
    def __init__(self, params):
        # TODO: hard-coded values
        self.pinhole_camera_intrinsics = PinholeCameraIntrinsics.from_params(params['env']['camera']['intrinsics'])
        camera_pos, camera_quat = get_camera_pose(np.array(params['env']['camera']['pos']),
                                                  np.array(params['env']['camera']['target_pos']),
                                                  np.array(params['env']['camera']['up_vector']))
        self.camera_pose = np.eye(4)
        self.camera_pose[0:3, 0:3] = camera_quat.rotation_matrix()
        self.camera_pose[0:3, 3] = camera_pos

        # params from yaml
        self.m_per_pixel = 265

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
    def get_heightmap(point_cloud, shape=(100, 100), grid_step=0.01):
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

    def extract_features(self, heightmap, target_size, plot=False):
        h, w = heightmap.shape
        cx = int(w / 2)
        cy = int(h / 2)

        bbox = ((target_size / 2.0) * self.m_per_pixel).astype(int)
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
        up_left_corners = [(int(cx - 16), int(cy - bbox[1] - 32)), (int(cx + bbox[0]), int(cy - 16)),
                           (int(cx - 16), int(cy + bbox[1])), (int(cx - bbox[0] - 32), int(cy - 16))]

        # Compute f_up, f_right, f_down, f_left
        for up_left_corner in up_left_corners:
            center = [up_left_corner[0] + 16, up_left_corner[1] + 16]
            cropped = heightmap[center[1] - 16:center[1] + 16,
                                center[0] - 16:center[0] + 16]
            features += Feature(cropped).pooling(kernel=[4, 4], stride=4).flatten().tolist()

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
                rect = patches.Rectangle(pt, 32, 32,
                                         linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            plt.show()

        return np.asarray(features)

    def state_representation(self, obs):
        # Get target pose from full state
        target = next(x for x in obs['full_state']['objects'] if x.name == 'target')

        # Get scene point cloud
        points = self.get_scene_points(obs)

        # Generate heightmap
        heightmap = self.get_heightmap(points)

        # Extract features
        features = self.extract_features(heightmap=heightmap, target_size=np.array([target.size[0], target.size[1]]))
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

    def reward(self, obs, next_obs, action):
        if next_obs['collision'] or empty_push(obs, next_obs):
            return -10.0

        # Singulation
        if len(self.point_around_target(next_obs)) == 0:
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
        elif len(self.point_around_target(next_obs)) == 0:
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
        distances_from_target = get_distances_from_target(obs)
        if all(dist > self.singulation_distance for dist in distances_from_target):
            return False
        return True
