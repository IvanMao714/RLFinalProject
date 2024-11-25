# import traci
# import numpy as np
# import random
# import timeit
# import os
# import tensorflow as tf
# from keras import Model
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.optimizers import Adam
#
# # phase codes based on environment.net.xml
# PHASE_NS_GREEN = 0  # action 0 code 00
# PHASE_NS_YELLOW = 1
# PHASE_NSL_GREEN = 2  # action 1 code 01
# PHASE_NSL_YELLOW = 3
# PHASE_EW_GREEN = 4  # action 2 code 10
# PHASE_EW_YELLOW = 5
# PHASE_EWL_GREEN = 6  # action 3 code 11
# PHASE_EWL_YELLOW = 7
#
# class ActorCriticSimulation:
#     def __init__(self, ActorModel, CriticModel, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
#         self._ActorModel = ActorModel
#         self._CriticModel = CriticModel
#         self._Memory = Memory
#         self._TrafficGen = TrafficGen
#         self._gamma = gamma
#         self._step = 0
#         self._sumo_cmd = sumo_cmd
#         self._max_steps = max_steps
#         self._green_duration = green_duration
#         self._yellow_duration = yellow_duration
#         self._num_states = num_states
#         self._num_actions = num_actions
#         self._reward_store = []
#         self._cumulative_wait_store = []
#         self._avg_queue_length_store = []
#         self._training_epochs = training_epochs
#
#     def run(self, episode):
#         start_time = timeit.default_timer()
#
#         self._TrafficGen.generate_routefile(seed=episode)
#         traci.start(self._sumo_cmd)
#         print("Simulating...")
#
#         # inits
#         self._step = 0
#         self._waiting_times = {}
#         self._sum_neg_reward = 0
#         self._sum_queue_length = 0
#         self._sum_waiting_time = 0
#         old_state = None
#         old_action = None
#         old_total_wait = 0
#
#         while self._step < self._max_steps:
#             current_state = self._get_state()
#
#             current_total_wait = self._collect_waiting_times()
#             # reward = old_total_wait - current_total_wait / 100
#             reward = -self._get_queue_length()
#
#             if old_state is not None:
#                 self._Memory.add_sample((old_state, old_action, reward, current_state))
#
#             action = self._choose_action(current_state)
#
#             if old_action is not None and old_action != action:
#                 self._set_yellow_phase(old_action)
#                 self._simulate(self._yellow_duration)
#
#             self._set_green_phase(action)
#             self._simulate(self._green_duration)
#
#             old_state = current_state
#             old_action = action
#             old_total_wait = current_total_wait
#
#             if reward < 0:
#                 self._sum_neg_reward += reward
#
#         self._save_episode_stats()
#         print("Total reward:", self._sum_neg_reward)
#         traci.close()
#         simulation_time = round(timeit.default_timer() - start_time, 1)
#
#         print("Training...")
#         start_time = timeit.default_timer()
#         for _ in range(self._training_epochs):
#             self._replay()
#         training_time = round(timeit.default_timer() - start_time, 1)
#
#         return simulation_time, training_time
#
#     def _simulate(self, steps_todo):
#         if (self._step + steps_todo) >= self._max_steps:
#             steps_todo = self._max_steps - self._step
#
#         while steps_todo > 0:
#             traci.simulationStep()
#             self._step += 1
#             steps_todo -= 1
#             queue_length = self._get_queue_length()
#             self._sum_queue_length += queue_length
#             self._sum_waiting_time += queue_length
#
#     def _collect_waiting_times(self):
#         incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
#         car_list = traci.vehicle.getIDList()
#         for car_id in car_list:
#             wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
#             road_id = traci.vehicle.getRoadID(car_id)
#             if road_id in incoming_roads:
#                 self._waiting_times[car_id] = wait_time
#             else:
#                 if car_id in self._waiting_times:
#                     del self._waiting_times[car_id]
#         total_waiting_time = sum(self._waiting_times.values())
#         return total_waiting_time
#
#     def _choose_action(self, state):
#         policy = self._ActorModel.predict(state).flatten()
#         action = np.random.choice(self._num_actions, p=policy)
#         return action
#
#     def _set_yellow_phase(self, old_action):
#         yellow_phase_code = old_action * 2 + 1
#         traci.trafficlight.setPhase("TL", yellow_phase_code)
#
#     def _set_green_phase(self, action_number):
#         if action_number == 0:
#             traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
#         elif action_number == 1:
#             traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
#         elif action_number == 2:
#             traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
#         elif action_number == 3:
#             traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)
#
#     def _get_queue_length(self):
#         halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
#         halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
#         halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
#         halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
#         queue_length = halt_N + halt_S + halt_E + halt_W
#         return queue_length
#
#     def _get_state(self):
#         state = np.zeros(self._num_states)
#         car_list = traci.vehicle.getIDList()
#
#         for car_id in car_list:
#             lane_pos = traci.vehicle.getLanePosition(car_id)
#             lane_id = traci.vehicle.getLaneID(car_id)
#             lane_pos = 750 - lane_pos
#
#             if lane_pos < 7:
#                 lane_cell = 0
#             elif lane_pos < 14:
#                 lane_cell = 1
#             elif lane_pos < 21:
#                 lane_cell = 2
#             elif lane_pos < 28:
#                 lane_cell = 3
#             elif lane_pos < 40:
#                 lane_cell = 4
#             elif lane_pos < 60:
#                 lane_cell = 5
#             elif lane_pos < 100:
#                 lane_cell = 6
#             elif lane_pos < 160:
#                 lane_cell = 7
#             elif lane_pos < 400:
#                 lane_cell = 8
#             elif lane_pos <= 750:
#                 lane_cell = 9
#
#             if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
#                 lane_group = 0
#             elif lane_id == "W2TL_3":
#                 lane_group = 1
#             elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
#                 lane_group = 2
#             elif lane_id == "N2TL_3":
#                 lane_group = 3
#             elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
#                 lane_group = 4
#             elif lane_id == "E2TL_3":
#                 lane_group = 5
#             elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
#                 lane_group = 6
#             elif lane_id == "S2TL_3":
#                 lane_group = 7
#             else:
#                 lane_group = -1
#
#             if lane_group >= 1 and lane_group <= 7:
#                 car_position = int(str(lane_group) + str(lane_cell))
#                 valid_car = True
#             elif lane_group == 0:
#                 car_position = lane_cell
#                 valid_car = True
#             else:
#                 valid_car = False
#
#             if valid_car:
#                 state[car_position] = 1
#
#         return state
#
#     def _replay(self):
#         batch = self._Memory.get_samples(self._ActorModel.batch_size)
#
#         if len(batch) > 0:
#             states = np.array([val[0] for val in batch])
#             actions = np.array([val[1] for val in batch])
#             rewards = np.array([val[2] for val in batch])
#             next_states = np.array([val[3] for val in batch])
#
#             # Predict V(s) and V(s')
#             values = self._CriticModel.predict(states).flatten()
#             next_values = self._CriticModel.predict(next_states).flatten()
#
#             # Compute TD targets and advantages
#             td_targets = rewards + self._gamma * next_values
#             advantages = td_targets - values
#
#             # Train critic
#             self._CriticModel.train(states, td_targets)
#
#             # Train actor
#             self._ActorModel.train(states, actions, advantages)
#
#     def _save_episode_stats(self):
#         self._reward_store.append(self._sum_neg_reward)
#         self._cumulative_wait_store.append(self._sum_waiting_time)
#         self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)
#
#     @property
#     def reward_store(self):
#         return self._reward_store
#
#     @property
#     def cumulative_wait_store(self):
#         return self._cumulative_wait_store
#
#     @property
#     def avg_queue_length_store(self):
#         return self._avg_queue_length_store
import traci
import numpy as np
import tensorflow as tf
import random
import timeit
import os

# 定义交通灯相位（根据您的环境）
PHASE_NS_GREEN = 0  # action 0
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3
PHASE_EWL_YELLOW = 7

class ActorCriticSimulation:
    def __init__(self, Agent, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Agent = Agent
        self._TrafficGen = TrafficGen
        # self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs
        self._waiting_times = {}

    def run(self, episode):
        """
        运行一次仿真，然后进行训练
        """
        start_time = timeit.default_timer()

        # 生成路线文件并启动 SUMO
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # 初始化
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = None
        old_action = None

        # while self._step < self._max_steps:
        #     # 获取当前状态
        #     current_state = self._get_state()
        #
        #     # 计算奖励
        #     current_total_wait = self._collect_waiting_times()
        #     reward = old_total_wait - current_total_wait
        #
        #     # 训练 Agent.py
        #     if old_state is not None and old_action is not None:
        #         self._Agent.train(old_state, old_action, reward, lr=0.001, para=[1, 1, False, 1, 1])
        #
        #     # 选择动作
        #     action = self._Agent.policy(current_state)
        #
        #     # 执行动作
        #     if old_action is not None and old_action != action:
        #         self._set_yellow_phase(old_action)
        #         self._simulate(self._yellow_duration)
        #
        #     self._set_green_phase(action)
        #     self._simulate(self._green_duration)
        #
        #     # 更新变量
        #     old_state = current_state
        #     old_action = action
        #     old_total_wait = current_total_wait
        #
        #     # 累积奖励
        #     if reward < 0:
        #         self._sum_neg_reward += reward
        #
        # # 仿真结束后，处理剩余的训练数据
        # self._Agent.train(current_state, action, reward, lr=0.001, para=[1, 1, True, 1, 1])
        while self._step < self._max_steps:
            current_state = self._get_state()

            # Calculate reward
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # Store experience
            if old_state is not None and old_action is not None:
                self._Agent.remember(old_state, old_action, reward)

            # Choose action
            action = self._Agent.policy(current_state)

            # Execute action
            if old_action is not None and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # Update variables
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # Accumulate negative reward
            if reward < 0:
                self._sum_neg_reward += reward

        # At the end of the simulation
        simulation_time = round(timeit.default_timer() - start_time, 1)
        start_time = timeit.default_timer()
        self._Agent.train(next_state=current_state, done=True)
        training_time = round(timeit.default_timer() - start_time, 1)
        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward)
        traci.close()

        return simulation_time, training_time

    def _simulate(self, steps_todo):
        """
        在 SUMO 中执行指定步数的仿真，并收集统计数据
        """
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length

    def _collect_waiting_times(self):
        """
        收集每辆车的等待时间
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times:
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _set_yellow_phase(self, old_action):
        """
        激活对应的黄色相位
        """
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        激活对应的绿色相位
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_queue_length(self):
        """
        获取排队车辆的数量
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _get_state(self):
        """
        获取当前交通状态
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # 使得靠近路口的位置为 0

            # 将位置映射到网格单元
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # 确定车道组
            if lane_id in ["W2TL_0", "W2TL_1", "W2TL_2"]:
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id in ["N2TL_0", "N2TL_1", "N2TL_2"]:
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id in ["E2TL_0", "E2TL_1", "E2TL_2"]:
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id in ["S2TL_0", "S2TL_1", "S2TL_2"]:
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 0 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))
                state[car_position] = 1

        return state

    def _save_episode_stats(self):
        """
        保存本次仿真的统计数据
        """
        self._reward_store.append(self._sum_neg_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store
