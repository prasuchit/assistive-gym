import itertools
from random import random
import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture

AGENT_TYPES = {
    0: 'Unfatigued',
    1: 'Fatigued',
    2: 'Collaborative',
    3: 'Super-Collaborative',
}

class TBFeedingEnv(AssistiveEnv):
    def __init__(self, robot, human):
        '''
        NOTE: PS: the obs_robot_len has been changed from 18 + len(controllable_joints) to 12 + len(controllable_joints) 
        because we removed 3 pose, 4 orient values of human head and added 1 self type value and 1 human type instead. 
        For human, the len went from 19 + len(joints) to 21 + len(joints))
        '''
        self.agent_names = ['robot', 'human']
        self.n_agents = len(self.agent_names)
        self.agent_types = AGENT_TYPES
        self.nAgent_type = len(self.agent_types)
        self.prev_agent_type = {'robot': 2, 'human': 0}
        self.current_energy_level = {'human': 10}
        self.curr_human_action = None
        self.belief_prob_counter = 0
        
        super(TBFeedingEnv, self).__init__(robot=robot, human=human, task='feeding', obs_robot_len=(13 + len(robot.controllable_joint_indices) -
                                            (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(21 + len(human.controllable_joint_indices)))

    def human_preferences(self, end_effector_velocity=0, total_force_on_human=0, tool_force_at_target=0, food_hit_human_reward=0, food_mouth_velocities=None, dressing_forces=None, arm_manipulation_tool_forces_on_human=None):
        
        if food_mouth_velocities is None:
            food_mouth_velocities = []
        if dressing_forces is None:
            dressing_forces = [[]]
        if arm_manipulation_tool_forces_on_human is None:
            arm_manipulation_tool_forces_on_human = [0, 0]
        
        # Slow end effector velocities
        reward_velocity = -end_effector_velocity

        # < 10 N force at target
        reward_high_target_forces = 0 if tool_force_at_target < 10 else -tool_force_at_target

        fatigued_human_action_penalty = 0
        # --- Scooping, Feeding, Drinking ---
        if self.task in ['feeding', 'drinking']:
            if self.prev_agent_type['human'] == 1:  # Human is fatigued
                action = np.copy(self.curr_human_action)
                fatigued_human_action_penalty = -np.mean(action)
            # Penalty when robot's body applies force onto a person
            reward_force_nontarget = -total_force_on_human
        # Penalty when robot spills food on the person
        reward_food_hit_human = food_hit_human_reward
        # Human prefers food entering mouth at low velocities
        reward_food_velocities = 0 if len(food_mouth_velocities) == 0 else -np.sum(food_mouth_velocities)

        return self.C_v*reward_velocity + self.C_f*reward_force_nontarget + self.C_hf*reward_high_target_forces + \
                self.C_fd*reward_food_hit_human + self.C_fdv*reward_food_velocities + fatigued_human_action_penalty


    def get_other_agents_types_ground_truth(self, curr_ag_id):
        '''
        @brief Given a particular agent id, returns a list of all the other agents' true type.
        '''
        return [
            self.prev_agent_type[ag_id]
            for ag_id in self.agent_names
            if ag_id != curr_ag_id
        ]
    
            
    def get_other_agents_types_for_belief_training(self, curr_ag_id):
        '''
        @brief Given a particular agent id, returns a list of all the other agents' probable type.
        This method is only used to train the belief network and make it learn the belief update distribution.
        This is not used during policy training or execution.
        '''
        # If human is asking for robot's type, always return groundtruth; if robot is asking for human's type
        # after a fatigued human has done 5 NoOp actions, always return groundtruth; otherwise return the type
        # fatigued or not with certain probability.
        return [
            self.prev_agent_type[ag_id] if ag_id == 0 or self.belief_prob_counter >= 5
            else random.choices([0, 1], weights=[(5 - self.belief_prob_counter) / 5, self.belief_prob_counter / 5], k=1)[0]
            for ag_id in self.agent_names if ag_id != curr_ag_id
        ]
        
    def update_types(self):
        '''
        @brief Updates energy levels of agents and if an agent's energy is 0 then updates their type.
        '''
        # Keep reducing human energy level at every step
        self.current_energy_level[self.agent_names[1]] -= 1 if self.current_energy_level[self.agent_names[1]] > 0 else self.current_energy_level[self.agent_names[1]]
        # If human energy level is totally depleted, change his type to fatigued
        if self.current_energy_level[self.agent_names[1]] == 0:
            self.prev_agent_type[self.agent_names[1]] = 1
        # If human is fatigued, robot becomes super-collaborative
        if self.prev_agent_type[self.agent_names[1]] == 1 and self.prev_agent_type[self.agent_names[0]] == 2:
            self.prev_agent_type[self.agent_names[0]] = 3

    def step(self, action):
        if self.human.controllable:
            self.curr_human_action = action['human']
            action = np.concatenate([action['robot'], action['human']])
            
        self.update_types()
        self.take_step(action)
        self.belief_prob_counter += 1
        
        obs = self._get_obs()

        reward_food, food_mouth_velocities, food_hit_human_reward = self.get_food_rewards()

        # Get human preferences
        end_effector_velocity = np.linalg.norm(
            self.robot.get_velocity(self.robot.right_end_effector))
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=self.total_force_on_human,
                                                    tool_force_at_target=self.spoon_force_on_human, food_hit_human_reward=food_hit_human_reward, food_mouth_velocities=food_mouth_velocities)

        spoon_pos, spoon_orient = self.tool.get_base_pos_orient()

        # Penalize robot for distance between the spoon and human mouth.
        reward_distance_mouth_target = - \
            np.linalg.norm(self.target_pos - spoon_pos)
            
        reward_action = -np.linalg.norm(action)  # Penalize actions

        reward = self.config('distance_weight')*reward_distance_mouth_target + self.config(
            'action_weight')*reward_action + self.config('food_reward_weight')*reward_food + preferences_score

        if self.gui and reward_food != 0:
            print('Task success:', self.task_success,
                    'Food reward:', reward_food)

        info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.total_food_count*self.config(
            'task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200 or self.task_success == 8

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}

    def get_total_force(self):
        robot_force_on_human = np.sum(
            self.robot.get_contact_points(self.human)[-1])
        spoon_force_on_human = np.sum(
            self.tool.get_contact_points(self.human)[-1])
        return robot_force_on_human, spoon_force_on_human

    def get_food_rewards(self):
        # Check all food particles to see if they have left the spoon or entered the person's mouth
        # Give the robot a reward or penalty depending on food particle status
        food_reward = 0
        food_hit_human_reward = 0
        food_mouth_velocities = []
        foods_to_remove = []
        foods_active_to_remove = []
        for f in self.foods:
            food_pos, food_orient = f.get_base_pos_orient()
            distance_to_mouth = np.linalg.norm(self.target_pos - food_pos)
            if distance_to_mouth < 0.03:
                # Food is close to the person's mouth. Delete particle and give robot a reward
                food_reward += 20
                self.task_success += 1
                food_velocity = np.linalg.norm(f.get_velocity(f.base))
                food_mouth_velocities.append(food_velocity)
                foods_to_remove.append(f)
                foods_active_to_remove.append(f)
                f.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                continue
            elif len(f.get_closest_points(self.tool, distance=0.1)[-1]) == 0:
                # Delete particle and give robot a penalty for spilling food
                food_reward -= 5
                foods_to_remove.append(f)
                continue
        for f in self.foods_active:
            if len(f.get_contact_points(self.human)[-1]) > 0:
                # Record that this food particle just hit the person, so that we can penalize the robot
                food_hit_human_reward -= 1
                foods_active_to_remove.append(f)
        self.foods = [f for f in self.foods if f not in foods_to_remove]
        self.foods_active = [
            f for f in self.foods_active if f not in foods_active_to_remove]
        return food_reward, food_mouth_velocities, food_hit_human_reward

    def _get_obs(self, agent=None):
        spoon_pos, spoon_orient = self.tool.get_base_pos_orient()
        spoon_pos_real, spoon_orient_real = self.robot.convert_to_realworld(
            spoon_pos, spoon_orient)
        robot_joint_angles = self.robot.get_joint_angles(
            self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (
            np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(
                self.robot.wheel_joint_indices):]
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        # head_pos_real, head_orient_real = self.robot.convert_to_realworld(head_pos, head_orient)
        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos)
        self.robot_force_on_human, self.spoon_force_on_human = self.get_total_force()
        self.total_force_on_human = self.robot_force_on_human + self.spoon_force_on_human
        # NOTE: PS: Removed head_pos_real, head_orient_real from robot obs to let human type provide the info that he's fatigued
        robot_obs = np.concatenate([spoon_pos_real, spoon_orient_real, spoon_pos_real - target_pos_real,
                                    robot_joint_angles, [self.spoon_force_on_human], [self.prev_agent_type[self.agent_names[0]]]]).ravel()
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(
                self.human.controllable_joint_indices)
            spoon_pos_human, spoon_orient_human = self.human.convert_to_realworld(
                spoon_pos, spoon_orient)
            head_pos_human, head_orient_human = self.human.convert_to_realworld(
                head_pos, head_orient)
            target_pos_human, _ = self.human.convert_to_realworld(
                self.target_pos)
            human_obs = np.concatenate([spoon_pos_human, spoon_orient_human, spoon_pos_human - target_pos_human, human_joint_angles,
                                        head_pos_human, head_orient_human, [self.robot_force_on_human, self.spoon_force_on_human], [self.prev_agent_type[self.agent_names[1]]]]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def reset(self):
        super(TBFeedingEnv, self).reset()
        self.build_assistive_env('wheelchair')
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient(
                wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, -np.pi/2.0])

        # Update robot and human motor gains
        self.robot.motor_gains = self.human.motor_gains = 0.025

        joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90),
                            (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        joints_positions += [(self.human.j_head_x, self.np_random.uniform(-30, 30)), (self.human.j_head_y,
                                                                                        self.np_random.uniform(-30, 30)), (self.human.j_head_z, self.np_random.uniform(-30, 30))]
        self.human.setup_joints(
            joints_positions, use_static_joints=True, reactive_force=None)

        # Create a table
        self.table = Furniture()
        self.table.init('table', self.directory, self.id, self.np_random)

        self.generate_target()

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45,
                                    cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)

        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory,
                        self.id, self.np_random, right=True, mesh_scale=[0.08]*3)

        target_ee_pos = np.array([-0.15, -0.65, 1.15]) + \
                self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(
            self.robot.toc_ee_orient_rpy[self.task])
        self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient), (self.target_pos, None)], [(
            self.target_pos, target_ee_orient)], arm='right', tools=[self.tool], collision_objects=[self.human, self.table, self.furniture])

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(
            self.robot.right_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        # Place a bowl on a table
        self.bowl = Furniture()
        self.bowl.init('bowl', self.directory, self.id, self.np_random)

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)
        self.tool.set_gravity(0, 0, 0)

        # p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)

        # Generate food
        spoon_pos, spoon_orient = self.tool.get_base_pos_orient()
        food_radius = 0.005
        food_mass = 0.001
        batch_positions = [
            np.array(
                [
                    i * 2 * food_radius - 0.005,
                    j * 2 * food_radius,
                    k * 2 * food_radius + 0.01,
                ]
            )
            + spoon_pos
            for i, j, k in itertools.product(range(2), range(2), range(2))
        ]
        self.foods = self.create_spheres(
            radius=food_radius, mass=food_mass, batch_positions=batch_positions, visual=False, collision=True)
        colors = [[60./256., 186./256., 84./256., 1], [244./256., 194./256., 13./256., 1],
                    [219./256., 50./256., 54./256., 1], [72./256., 133./256., 237./256., 1]]
        for i, f in enumerate(self.foods):
            p.changeVisualShape(
                f.body, -1, rgbaColor=colors[i % len(colors)], physicsClientId=self.id)
        self.total_food_count = len(self.foods)
        self.foods_active = list(self.foods)

        # Enable rendering
        p.configureDebugVisualizer(
            p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Drop food in the spoon
        for _ in range(25):
            p.stepSimulation(physicsClientId=self.id)

        self.init_env_variables()

        self.prev_agent_type = {'robot': 2,  'human': 0}
        self.current_energy_level = {'human': 10}
        self.time_elapsed = 0
        self.curr_human_action = None
        self.belief_prob_counter = 0

        return self._get_obs()

    def generate_target(self):
        # Set target on mouth
        self.mouth_pos = [
            0, -0.11, 0.03] if self.human.gender == 'male' else [0, -0.1, 0.03]
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        target_pos, target_orient = p.multiplyTransforms(
            head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target = self.create_sphere(
            radius=0.01, mass=0.0, pos=target_pos, collision=False, rgba=[0, 1, 0, 1])
        self.update_targets()

    def update_targets(self):
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        target_pos, target_orient = p.multiplyTransforms(
            head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])
