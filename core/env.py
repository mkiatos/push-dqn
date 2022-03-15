"""
Env
===

This module contains classes for defining an environment.
"""
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
import cv2


class Button:
    def __init__(self, title):
        self.id = p.addUserDebugParameter(title, 1, 0, 1)
        self.counter = p.readUserDebugParameter(self.id)
        self.counter_prev = self.counter

    def on(self):
        self.counter = p.readUserDebugParameter(self.id)
        if self.counter % 2 == 0:
            return True
        return False


class NamesButton(Button):
    def __init__(self, title):
        super(NamesButton, self).__init__(title)
        self.ids = []

    def show_names(self, objects):
        if self.on() and len(self.ids) == 0:
            for obj in objects:
                self.ids.append(p.addUserDebugText(text=obj.name, textPosition=[0, 0, 0], parentObjectUniqueId=obj.body_id))

        if not self.on():
            for i in self.ids:
                p.removeUserDebugItem(i)
            self.ids = []


class UR5Bullet:
    def __init__(self):
        self.num_joints = 6

        joint_names = ['ur5_shoulder_pan_joint', 'ur5_shoulder_lift_joint',
                       'ur5_elbow_joint', 'ur5_wrist_1_joint', 'ur5_wrist_2_joint',
                       'ur5_wrist_3_joint']

        self.camera_optical_frame = 'camera_color_optical_frame'
        self.ee_link_name = 'finger_tip'
        self.indices = bullet_util.get_joint_indices(joint_names, 0)

        self.joint_configs = {"home": [-2.8927932236757625, -1.7518461461930528, -0.8471216131631573,
                                       -2.1163833167682005, 1.5717067329577208, 0.2502483535771374],
                              "above_table": [-2.8964885089272934, -1.7541597533564786, -1.9212388653019141,
                                              -1.041716266062558, 1.5759665976832087, 0.24964880122853264]}

        self.reset_joint_position(self.joint_configs["home"])

        self.finger = [0.018, 0.018]

    def get_joint_position(self):
        joint_pos = []
        for i in range(self.num_joints):
            joint_pos.append(p.getJointState(0, self.indices[i])[0])
        return joint_pos

    def get_joint_velocity(self):
        joint_pos = []
        for i in range(self.num_joints):
            joint_pos.append(p.getJointState(0, self.indices[i])[1])
        return joint_pos

    def set_joint_position(self, joint_position):
        p.setJointMotorControlArray(bodyIndex=0,
                                    jointIndices=self.indices,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_position)

    def reset_joint_position(self, joint_position):
        for i in range(len(self.indices)):
            p.resetJointState(0, self.indices[i], joint_position[i])
        self.set_joint_position(joint_position)

    def get_task_pose(self):
        return bullet_util.get_link_pose(self.ee_link_name)

    def set_task_pose(self, pos, quat):
        link_index = bullet_util.get_link_indices([self.ee_link_name])[0]
        joints = p.calculateInverseKinematics(bodyIndex=0, endEffectorLinkIndex=link_index,
                                              targetPosition=(pos[0], pos[1], pos[2]),
                                              targetOrientation=quat.as_vector("xyzw"))
        self.set_joint_position(joints)

    def reset_task_pose(self, pos, quat):
        link_index = bullet_util.get_link_indices([self.ee_link_name])[0]
        joints = p.calculateInverseKinematics(bodyIndex=0, endEffectorLinkIndex=link_index,
                                              targetPosition=(pos[0], pos[1], pos[2]),
                                              targetOrientation=quat.as_vector("xyzw"))
        self.reset_joint_position(joints)

    def set_joint_trajectory(self, final, duration):
        init = self.get_joint_position()
        trajectories = []

        for i in range(self.num_joints):
            trajectories.append(Trajectory([0, duration], [init[i], final[i]]))

        t = 0
        dt = 1/240  # This is the dt of pybullet
        while t < duration:
            command = []
            for i in range(self.num_joints):
                command.append(trajectories[i].pos(t))
            self.set_joint_position(command)
            t += dt
            self.step()


class CameraBullet:
    def __init__(self, pos, target_pos, up_vector,
                 pinhole_camera_intrinsics, name='sim_camera'):
        self.name = name

        self.pos = np.array(pos)
        self.target_pos = np.array(target_pos)
        self.up_vector = np.array(up_vector)

        # Compute view matrix
        self.view_matrix = p.computeViewMatrix(cameraEyePosition=pos,
                                               cameraTargetPosition=target_pos,
                                               cameraUpVector=up_vector)

        self.z_near = 0.01
        self.z_far = 5.0
        self.width, self.height = pinhole_camera_intrinsics.width, pinhole_camera_intrinsics.height
        self.fx, self.fy = pinhole_camera_intrinsics.fx, pinhole_camera_intrinsics.fy
        self.cx, self.cy = pinhole_camera_intrinsics.cx, pinhole_camera_intrinsics.cy

        # Compute projection matrix
        fov_h = math.atan(self.height / 2 / self.fy) * 2 / math.pi * 180
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=fov_h, aspect=self.width / self.height,
                                                              nearVal=self.z_near, farVal=self.z_far)

    def get_pose(self):
        """
        Returns the camera pose w.r.t. world

        Returns
        -------
        np.array()
            4x4 matrix representing the camera pose w.r.t. world
        """
        return get_camera_pose(self.pos, self.target_pos, self.up_vector)

    def get_depth(self, depth_buffer):
        """
        Converts the depth buffer to depth map.

        Parameters
        ----------
        depth_buffer: np.array()
            The depth buffer as returned from opengl
        """
        depth = self.z_far * self.z_near / (self.z_far - (self.z_far - self.z_near) * depth_buffer)
        return depth

    def get_data(self):
        """
        Returns
        -------
        np.array(), np.array(), np.array()
            The rgb, depth and segmentation images
        """
        image = p.getCameraImage(self.width, self.height,
                                 self.view_matrix, self.projection_matrix,
                                 flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        return image[2], self.get_depth(image[3]), image[4]

    def get_intrinsics(self):
        """
        Returns the pinhole camera intrinsics
        """
        return PinholeCameraIntrinsics(width=self.width, height=self.height,
                                       fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)


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
    def __init__(self, robot, name='', params={}):
        super().__init__(name, params)
        self.render = params['render']
        if self.render:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            target = p.getDebugVisualizerCamera()[11]
            target_ = (target[0] - 0.5, target[1], target[2] + 0.3)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.85,
                cameraYaw=-70,
                cameraPitch=-45,
                cameraTargetPosition=target_)
        else:
            p.connect(p.DIRECT)
        self.params = params.copy()

        self.objects = []

        # Load table
        self.workspace_center_pos = np.array(params["workspace"]["pos"])
        self.workspace_center_quat = Quaternion(w=params["workspace"]["quat"]["w"],
                                                x=params["workspace"]["quat"]["x"],
                                                y=params["workspace"]["quat"]["y"],
                                                z=params["workspace"]["quat"]["z"])

        self.robot = None

        self.scene_generator = SceneGenerator(params['scene_generation'])

        # Set camera params
        pinhole_camera_intrinsics = PinholeCameraIntrinsics.from_params(params['camera']['intrinsics'])
        self.camera = CameraBullet(self.workspace2world(pos=params['camera']['pos'])[0],
                                   self.workspace2world(pos=params['camera']['target_pos'])[0],
                                   self.workspace2world(pos=params['camera']['up_vector'])[0],
                                   pinhole_camera_intrinsics)

        if self.render:
            self.button = Button("Pause")
            self.names_button = NamesButton("Show Names")
            self.slider = p.addUserDebugParameter("Delay sim (sec)", 0.0, 0.03, 0.0)
            self.exit_button = NamesButton("Exit")
        self.collision = False

        self.rng = np.random.RandomState()

    def load_robot_and_workspace(self):
        self.objects = []

        p.setAdditionalSearchPath("../assets")  # optionally

        # Load robot
        k = p.loadURDF("ur5e_rs_fingerlong.urdf")

        if self.params["workspace"]["walls"]:
            table_name = "table_walls.urdf"
        else:
            table_name = "table.urdf"

        table_id = p.loadURDF(table_name, basePosition=self.workspace_center_pos,
                              baseOrientation=self.workspace_center_quat.as_vector("xyzw"))

        # Todo: get table size w.r.t. local frame
        table_size = np.abs(np.asarray(p.getAABB(table_id)[1]) - np.asarray(p.getAABB(table_id)[0]))
        self.objects.append(Object(name='table', pos=self.workspace_center_pos,
                                   quat=self.workspace_center_quat.as_vector("xyzw"),
                                   size=(table_size[0], table_size[1]), body_id=table_id))

        # Load plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf", [0, 0, -0.7])
        self.objects.append(Object(name='plane', body_id=plane_id))

        self.robot = UR5Bullet()
        p.setJointMotorControlArray(bodyIndex=0, jointIndices=self.robot.indices, controlMode=p.POSITION_CONTROL)
        self.robot.reset_joint_position(self.robot.joint_configs["home"])

        pos, quat = self.workspace2world(pos=np.zeros(3), quat=Quaternion())
        scale = 0.3
        # p.addUserDebugLine(pos, pos + scale * quat.rotation_matrix()[:, 0], [1, 0, 0])
        # p.addUserDebugLine(pos, pos + scale * quat.rotation_matrix()[:, 1], [0, 1, 0])
        # p.addUserDebugLine(pos, pos + scale * quat.rotation_matrix()[:, 2], [0, 0, 1])

    def workspace2world(self, pos=None, quat=None, inv=False):
        """
        Transforms a pose in workspace coordinates to world coordinates

        Parameters
        ----------
        pos: list
            The position in workspace coordinates

        quat: Quaternion
            The quaternion in workspace coordinates

        Returns
        -------

        list: position in worldcreate_scene coordinates
        Quaternion: quaternion in world coordinates
        """
        world_pos, world_quat = None, None
        tran = Affine3.from_vec_quat(self.workspace_center_pos, self.workspace_center_quat).matrix()

        if inv:
            tran = Affine3.from_matrix(np.linalg.inv(tran)).matrix()

        if pos is not None:
            world_pos = np.matmul(tran, np.append(pos, 1))[:3]
        if quat is not None:
            world_rot = np.matmul(tran[0:3, 0:3], quat.rotation_matrix())
            world_quat = Quaternion.from_rotation_matrix(world_rot)

        return world_pos, world_quat

    def seed(self, seed=None):
        self.scene_generator.seed(seed)
        self.rng.seed(seed)

    def remove_body(self, idx):
        p.removeBody(id)
        obj = next(x for x in self.objects if x.body_id == idx)
        self.objects.remove(obj)
        self.scene_generator.objects.remove(obj)

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
        self.objects.append(single_obj)

    def add_boxes(self, obj):
        '''
        Parameters
        ----------
        obj
        obj_type 'box' or 'toyblock'
        '''
        crop_size = 193

        def get_pxl_distance(meters):
            return meters * crop_size / SURFACE_SIZE

        def get_xyz(pxl):
            x = min_max_scale(pxl[0], range=(0, 2 * 193), target_range=(-0.25, 0.25))
            y = -min_max_scale(pxl[1], range=(0, 2 * 193), target_range=(-0.25, 0.25))
            z = 0.02
            return np.array([x, y, z])

        for i in range(len(obj)):
            max_size = np.sqrt(obj[i].size[0] ** 2 + obj[i].size[1] ** 2)
            erode_size = int(np.round(get_pxl_distance(meters=max_size)))
            seg = self.get_obs()['seg']
            seg = Feature(seg).crop(crop_size, crop_size).array()
            free = np.zeros(seg.shape, dtype=np.uint8)
            free[seg == 1] = 1
            free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
            free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
            if np.sum(free) == 0:
                return

            if i == 0:
                pix = np.array([free.shape[0] / 2.0, free.shape[1] / 2.0])
            else:
                pixx = util.math.sample_distribution(np.float32(free))
                pix = np.array([pixx[1], pixx[0]])

            plt.imshow(free)
            plt.plot(pix[0], pix[1], 'ro')
            plt.show()

            obj[i].pos = get_xyz(pix)
            theta = self.rng.rand() * 2 * np.pi
            obj[i].quat = Quaternion().from_rotation_matrix(util.orientation.rot_z(theta))
            self.add_single_box(obj[i])

    def reset(self):
        self.collision = False
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        self.load_robot_and_workspace()

        hugging = self.rng.random() < self.params['scene_generation']['hug']['probability']
        self.scene_generator.reset()

        self.scene_generator.generate_scene(hugging=hugging)

        if self.params['scene_generation']['object_type'] == 'boxes':
            self.add_boxes(self.scene_generator.objects)
        elif self.params['scene_generation']['object_type'] == 'toyblocks':
            self.add_toyblocks(self.scene_generator.objects, hugging)
            # self.add_challenging(self.scene_generator, test_preset_file='../assets/test-cases/challenging_scene_8.txt')
        else:
            raise AttributeError

        # Uncomment for old scene generation
        # for obj in self.scene_generator.objects:
        #     self.add_box(obj)

        t = 0
        while t < 3000:
            p.stepSimulation()
            t += 1

        # Update position and orientation
        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            obj.pos, obj.quat = self.workspace2world(pos=np.array(pos),
                                                     quat=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                                                     inv=True)

        if hugging:
            self.hug()

        t = 0
        while self.objects_still_moving():
            time.sleep(0.001)
            self.sim_step()
            t += 1
            if t > 3000:
                self.reset()


        # self.clear_target_occlusion()

        return self.get_obs()

    def objects_still_moving(self):
        for obj in self.objects:
            if obj.name in ['table', 'plane']:
                continue

            vel, rot_vel = p.getBaseVelocity(bodyUniqueId=obj.body_id)
            norm_1 = np.linalg.norm(vel)
            norm_2 = np.linalg.norm(rot_vel)
            if norm_1 > 0.005 or norm_2 > 0.3:
                return True
        return False

    def step(self, action):
        if len(action) > 2:

            self.collision = False
            p1 = action[0]
            p2 = action[-1]
            p1_w, _ = self.workspace2world(p1)
            p2_w, _ = self.workspace2world(p2)

            tmp_1 = p1_w.copy()
            tmp_2 = p2_w.copy()
            tmp_1[2] = 0
            tmp_2[2] = 0
            y_direction = (tmp_2 - tmp_1) / np.linalg.norm(tmp_2 - tmp_1)
            x = np.cross(y_direction, np.array([0, 0, -1]))

            rot_mat = np.array([[x[0], y_direction[0], 0],
                                [x[1], y_direction[1], 0],
                                [x[2], y_direction[2], -1]])

            quat = Quaternion.from_rotation_matrix(rot_mat)

            # Inverse kinematics seems to not accurate when the target position is far from the current,
            # resulting to errors after reset. Call trajectory to compensate for these errors
            self.robot.reset_task_pose(p1_w + np.array([0, 0, 0.05]), quat)
            self.robot_set_task_pose_trajectory(p1_w + np.array([0, 0, 0.05]), quat, 0.2)
            self.robot_set_task_pose_trajectory(p1_w, quat, 0.5, stop_collision=True)

            if not self.collision:
                for i in range(1, len(action)):
                    p_w, _ = self.workspace2world(action[i])
                    self.robot_set_task_pose_trajectory(p_w, quat, None)
                # self.robot_set_task_pose_trajectory(p2_w + np.array([0, 0, 0.05]), quat, 1)

            self.robot.reset_joint_position(self.robot.joint_configs["home"])

            while self.objects_still_moving():
                time.sleep(0.001)
                self.sim_step()

            return self.get_obs()
        else:
            return self.step_linear(action)

    def step_linear(self, action):
        """
        Moves the environment one step forward in time.

        Parameters
        ----------

        action : tuple
            A tuple of two 3D np.arrays corresponding to the initial and final 3D point of the push with respect to
            inertia frame (workspace frame)

        Returns
        -------
        dict :
            A dictionary with the following keys: rgb, depth, seg, full_state. See get_obs() for more info.
        """
        self.collision = False
        p1 = action[0]
        p2 = action[1]
        p1_w, _ = self.workspace2world(p1)
        p2_w, _ = self.workspace2world(p2)

        y_direction = (p2_w - p1_w) / np.linalg.norm(p2_w - p1_w)
        x = np.cross(y_direction, np.array([0, 0, -1]))

        rot_mat = np.array([[x[0], y_direction[0], 0],
                            [x[1], y_direction[1], 0],
                            [x[2], y_direction[2], -1]])

        quat = Quaternion.from_rotation_matrix(rot_mat)

        # Inverse kinematics seems to not accurate when the target position is far from the current,
        # resulting to errors after reset. Call trajectory to compensate for these errors
        self.robot.reset_task_pose(p1_w + np.array([0, 0, 0.05]), quat)
        self.robot_set_task_pose_trajectory(p1_w + np.array([0, 0, 0.05]), quat, 0.2)
        self.robot_set_task_pose_trajectory(p1_w, quat, 0.5, stop_collision=True)

        if not self.collision:
            self.robot_set_task_pose_trajectory(p2_w, quat, None)
        # self.robot_set_task_pose_trajectory(p2_w + np.array([0, 0, 0.05]), quat, 1)

        self.robot.reset_joint_position(self.robot.joint_configs["home"])

        while self.objects_still_moving():
            time.sleep(0.001)
            self.sim_step()

        return self.get_obs()

    def get_obs(self):
        # Update visual observation
        rgb, depth, seg = self.camera.get_data()

        # Update position and orientation
        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            obj.pos, obj.quat = self.workspace2world(pos=np.array(pos),
                                                     quat=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                                                     inv=True)

        table = next(x for x in self.objects if x.name == 'table')
        full_state = {'objects': self.objects,
                      'finger': self.robot.finger,
                      'surface': [table.size[0], table.size[1]]}

        import copy
        return {'rgb': rgb, 'depth': depth, 'seg': seg, 'full_state': copy.deepcopy(full_state),
                'collision': self.collision}

    def hug(self):
        target = next(x for x in self.objects if x.name == 'target')
        ids = []
        forces = {'boxes': 20, 'toyblocks': 15}
        force_magnitude = forces[self.params['scene_generation']['object_type']]
        duration = 300
        t = 0
        while t < duration:
            for obj in self.objects:
                if obj.name in ['table', 'plane']:
                    continue

                if obj.pos[2] < 0:
                    continue

                if obj.name == 'target':
                    force_magnitude = 3 * forces[self.params['scene_generation']['object_type']]
                else:
                    force_magnitude = forces[self.params['scene_generation']['object_type']]

                pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
                error = self.workspace2world(target.pos)[0] - pos
                if np.linalg.norm(error) < self.params['scene_generation']['hug']['radius']:
                    force_direction = error / np.linalg.norm(error)
                    pos_apply = np.array([pos[0], pos[1], 0])
                    p.applyExternalForce(obj.body_id, -1, force_magnitude * force_direction, pos_apply, p.WORLD_FRAME)

            p.stepSimulation()
            t += 1

        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            obj.pos, obj.quat = self.workspace2world(pos=np.array(pos),
                                                     quat=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                                                     inv=True)

    def robot_set_task_pose_trajectory(self, pos, quat, duration, stop_collision=False):
        init_pos, init_quat = self.robot.get_task_pose()
        # Calculate duration adaptively if its None
        if duration is None:
            vel = 0.3
            duration = np.linalg.norm(init_pos - pos) / vel
        trajectories = []
        for i in range(3):
            trajectories.append(Trajectory([0, duration], [init_pos[i], pos[i]]))

        t = 0
        dt = 1/240  # This is the dt of pybullet
        while t < duration:
            command = []
            for i in range(3):
                command.append(trajectories[i].pos(t))
            self.robot.set_task_pose(command, init_quat)
            t += dt
            contact = self.sim_step()

            if stop_collision and contact:
                self.collision = True
                break

    def sim_step(self):
        if self.render:
            if self.exit_button.on():
                exit()

            while self.button.on():
                time.sleep(0.001)

            # time.sleep(p.readUserDebugParameter(self.slider))
            time.sleep(0.005)
        p.stepSimulation()

        if self.render:
            self.names_button.show_names(self.objects)

        link_index = bullet_util.get_link_indices(['finger_body'])[0]

        contact = False
        for obj in self.objects:
            if obj.name == 'table' or obj.name == 'plane':
                continue

            contacts = p.getContactPoints(0, obj.body_id, link_index, -1)
            valid_contacts = []
            for c in contacts:
                normal_vector = c[7]
                normal_force = c[9]
                if np.dot(normal_vector, np.array([0, 0, 1])) > 0.9:
                    valid_contacts.append(c)

            if len(valid_contacts) > 0:
                contact = True
                break

        return contact
