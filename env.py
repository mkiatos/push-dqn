"""
Env
===

This module contains classes for defining an environment.
"""
import numpy as np
import pybullet as p
import pybullet_data
import time
import cv2
import matplotlib.pyplot as plt

from clt_core.env import Env
import clt_core as clt

SURFACE_SIZE = 0.25


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

        distance = clt.bullet_util.get_distance_of_two_bbox(target_pose, target.size, obj_pose, obj.size)

        # points = p.getClosestPoints(target.body_id, obj.body_id, distance=10)
        # distance = np.linalg.norm(np.array(points[0][5]) - np.array(points[0][6]))

        distances_from_target.append(distance)
    return np.array(distances_from_target)


class BulletEnv(Env):
    """
    Class implementing the clutter env in pyBullet.

    Parameters
    ----------
    name : str
        A string with the name of the environment.
    params : dict
        A dictionary with parameters for the environment.
    """
    def __init__(self, name='', params={}):
        super(BulletEnv, self).__init__(name, params)

    def load_robot_and_workspace(self):
        self.objects = []

        p.setAdditionalSearchPath("assets")  # optionally

        # Load robot
        robot_id = p.loadURDF("ur5e_rs_fingerlong.urdf")

        table_id = p.loadURDF("table.urdf", basePosition=self.workspace_center_pos,
                              baseOrientation=self.workspace_center_quat.as_vector("xyzw"))

        # Todo: get table size w.r.t. local frame
        table_size = np.abs(np.asarray(p.getAABB(table_id)[1]) - np.asarray(p.getAABB(table_id)[0]))
        self.objects.append(clt.core.Object(name='table', pos=self.workspace_center_pos,
                                            quat=self.workspace_center_quat.as_vector("xyzw"),
                                            size=(table_size[0], table_size[1]), body_id=table_id))

        # Load plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf", [0, 0, -0.7])
        self.objects.append(clt.core.Object(name='plane', body_id=plane_id))

        self.robot = clt.env.UR5Bullet(robot_id)
        p.setJointMotorControlArray(bodyIndex=0, jointIndices=self.robot.indices, controlMode=p.POSITION_CONTROL)
        self.robot.reset_joint_position(self.robot.joint_configs["home"])

        pos, quat = self.workspace2world(pos=np.zeros(3), quat=clt.ori.Quaternion())
        scale = 0.3
        p.addUserDebugLine(pos, pos + scale * quat.rotation_matrix()[:, 0], [1, 0, 0])
        p.addUserDebugLine(pos, pos + scale * quat.rotation_matrix()[:, 1], [0, 1, 0])
        p.addUserDebugLine(pos, pos + scale * quat.rotation_matrix()[:, 2], [0, 0, 1])

        self.target_mask = []

    def add_single_box(self, single_obj):
        if single_obj.name == 'target':
            color = [1, 0, 0, 1]
        else:
            color = [0, 0, 1, 1]
        col_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=single_obj.size)
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                              halfExtents=single_obj.size,
                                              rgbaColor=color)

        pos = single_obj.pos
        if single_obj.name != 'target':
            pos[2] += 0.03

        base_position, base_orientation = self.workspace2world(pos=single_obj.pos, quat=single_obj.quat)
        base_orientation = base_orientation.as_vector("xyzw")
        mass = 1.0
        body_id = p.createMultiBody(mass, col_box_id, visual_shape_id,
                                    base_position, base_orientation)
        single_obj.body_id = body_id
        return body_id

    def add_boxes(self):
        '''
        Parameters
        ----------
        obj
        '''
        crop_size = 193

        def get_pxl_distance(meters):
            return meters * crop_size / SURFACE_SIZE

        def get_xyz(pxl):
            x = clt.math.min_max_scale(pxl[0], range=(0, 2 * 193), target_range=(-0.25, 0.25))
            y = -clt.math.min_max_scale(pxl[1], range=(0, 2 * 193), target_range=(-0.25, 0.25))
            z = 0.02
            return np.array([x, y, z])

        def sample_size(min_bounding_box, max_bounding_box):
            a = min_bounding_box[2]
            b = max_bounding_box[2]
            if a > b:
                b = a
            target_height = self.rng.uniform(a, b)

            a = min_bounding_box[0]
            b = max_bounding_box[0]
            if a > b:
                b = a
            target_length = self.rng.uniform(a, b)

            a = min_bounding_box[1]
            b = min(target_length, max_bounding_box[1])
            if a > b:
                b = a
            target_width = self.rng.uniform(a, b)
            size = [target_length, target_width, target_height]
            return size

        def sample_boxes():
            nr_of_obstacles = self.params['scene_generation']['nr_of_obstacles']
            n_obstacles = nr_of_obstacles[0] + self.rng.randint(nr_of_obstacles[1] - nr_of_obstacles[0] + 1)

            target_size = sample_size(self.params['scene_generation']['target']['min_bounding_box'],
                                      self.params['scene_generation']['target']['max_bounding_box'])
            objects = [clt.core.Object(name='target', pos=np.array([1.0, 1.0, 0.05]), quat=clt.ori.Quaternion(),
                                       size=target_size)]

            for j in range(n_obstacles):
                obs_size = sample_size(self.params['scene_generation']['obstacle']['min_bounding_box'],
                                      self.params['scene_generation']['obstacle']['max_bounding_box'])

                obj = clt.core.Object(name='obs_' + str(j),
                                      pos=np.array([1.0, 1.0, 0.05]),
                                      quat=clt.core.Quaternion(),
                                      size=obs_size)
                objects.append(obj)
            return objects

        # Sample n objects from the database.
        objs = sample_boxes()

        for i in range(len(objs)):
            body_id = self.add_single_box(objs[i])
            p.removeBody(body_id)

            max_size = np.sqrt(objs[i].size[0] ** 2 + objs[i].size[1] ** 2)
            erode_size = int(np.round(get_pxl_distance(meters=max_size)))
            seg = super(BulletEnv, self).get_obs()['seg']
            seg = clt.cv_tools.Feature(seg).crop(crop_size, crop_size).array()
            free = np.zeros(seg.shape, dtype=np.uint8)
            free[seg == 1] = 1
            free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
            free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
            if np.sum(free) == 0:
                return

            if i == 0:
                pix = np.array([free.shape[0] / 2.0, free.shape[1] / 2.0])
            else:
                pixx = clt.math.sample_distribution(np.float32(free), rng=self.rng)
                pix = np.array([pixx[1], pixx[0]])

            # plt.imshow(free)
            # plt.plot(pix[0], pix[1], 'ro')
            # plt.show()

            objs[i].pos = get_xyz(pix)
            theta = self.rng.rand() * 2 * np.pi
            theta = 0
            objs[i].quat = clt.ori.Quaternion().from_rotation_matrix(clt.ori.rot_z(theta))
            body_id = self.add_single_box(objs[i])
            self.objects.append(clt.core.Object(name=objs[i].name, pos=objs[i].pos, quat=objs[i].quat,
                                                size=objs[i].size, body_id=body_id))

            if i == 0:
                self.target_mask = super(BulletEnv, self).get_obs()['seg']
                # plt.imshow(seg)
                # plt.show()

    def reset(self):
        self.collision = False
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        self.load_robot_and_workspace()

        self.add_boxes()

        t = 0
        while t < 3000:
            p.stepSimulation()
            t += 1

        # Update position and orientation
        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            obj.pos, obj.quat = self.workspace2world(pos=np.array(pos),
                                                     quat=clt.ori.Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                                                     inv=True)

        self.hug()

        t = 0
        while self.objects_still_moving():
            time.sleep(0.001)
            self.sim_step()
            t += 1
            if t > 3000:
                self.reset()

        return self.get_obs()

    def hug(self, force=8, duration=300):
        target = next(x for x in self.objects if x.name == 'target')
        t = 0
        while t < duration:
            for obj in self.objects:
                if obj.name in ['table', 'plane']:
                    continue

                if obj.pos[2] < 0:
                    continue

                if obj.name == 'target':
                    force_magnitude = 3 * force
                else:
                    force_magnitude = force

                pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
                error = self.workspace2world(target.pos)[0] - pos
                if np.linalg.norm(error) < 1e-6:
                    force_direction = np.array([0, 0, 0])
                else:
                    force_direction = error / np.linalg.norm(error)
                pos_apply = np.array([pos[0], pos[1], 0])
                p.applyExternalForce(obj.body_id, -1, force_magnitude * force_direction, pos_apply, p.WORLD_FRAME)

            p.stepSimulation()
            t += 1

        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            obj.pos, obj.quat = self.workspace2world(pos=np.array(pos),
                                                     quat=clt.ori.Quaternion(x=quat[0], y=quat[1], z=quat[2],
                                                                             w=quat[3]),
                                                     inv=True)

    def get_obs(self):
        obs = super(BulletEnv, self).get_obs()
        obs['target_mask'] = self.target_mask
        return obs

