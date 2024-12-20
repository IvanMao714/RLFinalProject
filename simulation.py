import random
import timeit

import numpy as np
import torch
import traci
from soupsieve import select

from RLfinal.envir.generator import TrafficGenerator
from RLfinal.memory import Memory

# Phase codes for controlling traffic lights based on the environment's net.xml file
PHASE_NS_GREEN = 0  # Action 0: North-South straight green
PHASE_NS_YELLOW = 1  # North-South straight yellow
PHASE_NSL_GREEN = 2  # Action 1: North-South left-turn green
PHASE_NSL_YELLOW = 3  # North-South left-turn yellow
PHASE_EW_GREEN = 4  # Action 2: East-West straight green
PHASE_EW_YELLOW = 5  # East-West straight yellow
PHASE_EWL_GREEN = 6  # Action 3: East-West left-turn green
PHASE_EWL_YELLOW = 7  # East-West left-turn yellow

class Training_Simulation:
    def __init__(self, config, sumo_cmd, agent, memory, model_name):

        """
        Initialize the training simulation class.

        Args:
            config (dict): Configuration dictionary containing parameters like max_steps, durations, etc.
            sumo_cmd (list): Command to initialize the SUMO simulation.
            agent: RL agent for decision-making (DQN, SAC, or Q-learning).
            memory: Replay memory for storing transitions.
            model_name (str): Name of the RL model being trained.
        """
        # Initialize agent and memory
        self._Memory = memory
        self._Agent =  agent

        # Traffic and simulation parameters
        self._TrafficGen = TrafficGenerator(config['max_steps'],config['n_cars_generated'])
        self.epsilon = config['epsilon']
        self._gamma = config['gamma']
        self._max_steps = config['max_steps']
        self._green_duration = config['green_duration']
        self._yellow_duration = config['yellow_duration']
        self._num_states = config['num_states']
        self._num_actions = config['num_actions']
        self._training_epochs = config['training_epochs']
        self._batch_size = config['batch_size']

        # Simulation tracking variables
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []

        # Epsilon-greedy exploration parameters
        self._epsilon = config['epsilon']

        self.model_name = model_name


    def run(self, episode):
        """
        Run an episode of the simulation, updating Q-table or neural networks during training.

        Args:
            episode (int): Current episode index.

        Returns:
            simulation_time (float): Time taken for the simulation.
            training_time (float): Time taken for training the agent.
        """
        # Q-learning specific logic
        if self.model_name == 'Q-learning':
            start_time = timeit.default_timer()

            # Generate route file and start SUMO simulation

            self._TrafficGen.generate_routefile(seed=episode)
            traci.start(self._sumo_cmd)
            print("Simulating...")

            # Initialize episode variables

            self._step = 0
            self._waiting_times = {}
            self._sum_neg_reward = 0
            self._sum_queue_length = 0
            self._sum_waiting_time = 0
            old_total_wait = 0

            old_state = None
            old_action = None

            while self._step < self._max_steps:
                # Get the current state as a one-hot vector
                current_state = self._get_state()
                # Convert state to integer index
                current_s = int(np.argmax(current_state))

                # Calculate reward based on waiting times
                current_total_wait = self._collect_waiting_times()
                # reward = (old_total_wait - current_total_wait) if self._step > 0 else 0
                reward = -(current_total_wait / max(len(self._waiting_times), 1)) if self._step > 0 else 0

                # Update Q-table based on the previous state, action, and reward
                if self._step > 0:
                    
                    old_s = int(np.argmax(old_state))
                    a = old_action
                    r = reward
                    s_next = current_s
                    dw = (self._step >= self._max_steps - 1)  

                    
                    self._Agent.train(old_s, a, r, s_next, dw)

                # Select the next action using epsilon-greedy policy
                action = self._choose_action(current_state)

                # Transition to yellow phase if the action changes
                if self._step > 0 and old_action is not None and old_action != action:
                    self._set_yellow_phase(old_action)
                    self._simulate(self._yellow_duration)

                # Execute the green phase for the chosen action
                self._set_green_phase(action)
                self._simulate(self._green_duration)

                ######################debug#############
                # print(
                #     f"Step: {self._step}, Reward: {reward}, Old Total Wait: {old_total_wait}, Current Total Wait: {current_total_wait}")

                # Update tracking variables

                old_state = current_state
                old_action = action
                old_total_wait = current_total_wait

                

                self._sum_neg_reward += reward

            self._save_episode_stats()
            print("Total reward:", self._sum_neg_reward)
            traci.close()
            simulation_time = round(timeit.default_timer() - start_time, 1)

            print("Training...")
            start_time = timeit.default_timer()

            
            # for _ in range(self._training_epochs):
            #     self._Agent.train() # 不需要

            training_time = round(timeit.default_timer() - start_time, 1)
        else:
            start_time = timeit.default_timer()

            # first, generate the route file for this simulation and set up sumo
            self._TrafficGen.generate_routefile(seed=episode)
            traci.start(self._sumo_cmd)
            print("Simulating...")

            # inits
            self._step = 0
            self._waiting_times = {}
            self._sum_neg_reward = 0
            self._sum_queue_length = 0
            self._sum_waiting_time = 0
            old_total_wait = 0
            old_state = -1
            old_action = -1
            count=0
            while self._step < self._max_steps:
                # get current state of the intersection
                current_state = self._get_state()

                # calculate reward of previous action: (change in cumulative waiting time between actions)
                # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
                current_total_wait = self._collect_waiting_times()
                reward = old_total_wait - current_total_wait
                # reward = -current_total_wait

                # saving the data into the memory
                if self._step != 0:
                    self._Memory.add_sample((old_state, old_action, reward, current_state))

                # choose the light phase to activate, based on the current state of the intersection
                action = self._choose_action(current_state)

                # if the chosen phase is different from the last phase, activate the yellow phase
                if self._step != 0 and old_action != action:
                    self._set_yellow_phase(old_action)
                    self._simulate(self._yellow_duration)

                # execute the phase selected before
                self._set_green_phase(action)
                self._simulate(self._green_duration)

                # saving variables for later & accumulate reward
                old_state = current_state
                old_action = action
                old_total_wait = current_total_wait

                # saving only the meaningful reward to better see if the agent is behaving correctly
                # if reward < 0:
                self._sum_neg_reward += reward

            self._save_episode_stats()
            print("Total reward:", self._sum_neg_reward)
            traci.close()
            simulation_time = round(timeit.default_timer() - start_time, 1)

            print("Training...")
            start_time = timeit.default_timer()
            # for _ in range(self._training_epochs):
            #     self._replay()

            for _ in range(self._training_epochs):
                # print(self._Memory.size_now())
                self._Agent.train()
            training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time



    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds

    def _choose_action(self, state):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        # if random.random() < self._epsilon:
        #     return random.randint(0, self._num_actions - 1)  # random action
        # else:
        #     return np.argmax(self._Agent.predict_one(state))  # the best action given the current state
        return self._Agent.select_action(state, False)

    # utils function

    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in SUMO.

        **Inputs**:
        - `action_number` (int): The index of the desired green phase.
            - `0`: Activate North-South green (PHASE_NS_GREEN).
            - `1`: Activate North-South left-turn green (PHASE_NSL_GREEN).
            - `2`: Activate East-West green (PHASE_EW_GREEN).
            - `3`: Activate East-West left-turn green (PHASE_EWL_GREEN).

        **Outputs**:
        - None. This function interacts with SUMO via `traci` to set the traffic light phase to green.

        **Details**:
        - Based on `action_number`, the function maps the input to the corresponding green phase defined in the SUMO configuration.
        - The traffic light phase is updated using:
          ```python
          traci.trafficlight.setPhase("TL", green_phase_code)
          ```
          where `green_phase_code` is one of:
            - `PHASE_NS_GREEN`: Green for North-South straight movement.
            - `PHASE_NSL_GREEN`: Green for North-South left turns.
            - `PHASE_EW_GREEN`: Green for East-West straight movement.
            - `PHASE_EWL_GREEN`: Green for East-West left turns.

        **Example**:
        - If `action_number = 0`, the function will:
          - Set the traffic light to `PHASE_NS_GREEN`.
        - If `action_number = 3`, the function will:
          - Set the traffic light to `PHASE_EWL_GREEN`.
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)  # North-South straight green
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)  # North-South left-turn green
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)  # East-West straight green
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)  # East-West left-turn green

    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in SUMO.

        **Inputs**:
        - `old_action` (int): The index of the traffic light phase before transitioning to yellow.
            - Valid values depend on the number of phases in the traffic light system.
            - Typically, `old_action` corresponds to the index of the last active green light phase.

        **Outputs**:
        - None. This function interacts with SUMO via `traci` to set the traffic light phase to yellow.

        **Details**:
        - The yellow phase code is calculated as `old_action * 2 + 1`.
          - This logic assumes that yellow phases are interleaved between green phases in the traffic light configuration defined in `environment.net.xml`.
          - For example:
            - If `old_action = 0`, yellow phase = `1`.
            - If `old_action = 1`, yellow phase = `3`.
        - The traffic light phase is updated using:
          ```python
          traci.trafficlight.setPhase("TL", yellow_phase_code)
          ```
          where `"TL"` is the traffic light ID.

        **Example**:
        - If the current green phase corresponds to `old_action = 2`, the function will:
          - Calculate `yellow_phase_code = 2 * 2 + 1 = 5`.
          - Set the traffic light to phase `5` (yellow).
        """
        yellow_phase_code = old_action * 2 + 1  # Calculate the yellow phase code
        traci.trafficlight.setPhase("TL", yellow_phase_code)  # Set the yellow phase

    def _get_state(self):
        """
        Retrieve the state of the intersection from SUMO, in the form of cell occupancy.

        **Inputs**:
        - This function does not take any explicit parameters but relies on the following attributes of the class:
            - `self._num_states` (int): Total number of states (cells) representing the intersection.
            - SUMO's `traci.vehicle` module to retrieve real-time vehicle information.

        **Outputs**:
        - `state` (np.ndarray): A 1D numpy array of size `self._num_states` (typically 80 or more). Each element is:
            - `1` if the corresponding cell is occupied by a vehicle.
            - `0` if the cell is unoccupied.

        **Function Details**:
        - Fetches a list of all vehicle IDs currently in the simulation using `traci.vehicle.getIDList()`.
        - For each vehicle:
            1. Calculates its relative position to the traffic light (lane position).
            2. Maps this position into one of several discrete "cells" along the lane, based on distance ranges.
            3. Determines which lane group the vehicle is in (e.g., westbound, northbound, etc.), including specific turn-only lanes.
            4. Combines the lane group and cell ID to create a unique "car position" identifier.
        - Updates the `state` array by marking the cell as occupied (`1`) if the vehicle is in a valid lane group and within the intersection's area of interest.

        **Key Assumptions**:
        - Lane positions range from `0` (near the traffic light) to `750` (far end of the road).
        - Lane IDs follow a specific naming convention (e.g., "W2TL_0" for west-to-traffic-light lanes).
        - Vehicles crossing or exiting the intersection are ignored.

        **Example**:
        Suppose there are 2 vehicles:
        - Vehicle 1 in lane "W2TL_0" at position 5m from the traffic light.
        - Vehicle 2 in lane "E2TL_1" at position 50m from the traffic light.

        The function will:
        - Map Vehicle 1 to lane group `0`, cell `0`.
        - Map Vehicle 2 to lane group `4`, cell `5`.
        - Return a `state` array where `state[0] = 1` and `state[45] = 1` (assuming 45 = lane group `4` concatenated with cell `5`).

        **Returns**:
        - A numpy array representing the current state of the intersection in terms of cell occupancy.
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

            # distance in meters from the traffic light -> mapping into cells
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

            # finding the lane where the car is located
            # x2TL_3 are the "turn left only" lanes
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))  # composition of the two position IDs to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        return state

    def _save_episode_stats(self):
        """
        Save the statistics of the episode for analysis and visualization.

        **Inputs**:
        - This method does not take any explicit arguments but relies on the following class attributes:
            - `self._sum_neg_reward` (float): The cumulative negative reward collected during the episode, representing penalties.
            - `self._sum_waiting_time` (float): The total waiting time (in seconds) for all cars in the episode.
            - `self._sum_queue_length` (int): The total queue length (number of cars waiting) summed over all steps in the episode.
            - `self._max_steps` (int): The total number of simulation steps in the episode.
            - `self._reward_store` (list): A list to store the cumulative negative reward for each episode.
            - `self._cumulative_wait_store` (list): A list to store the total waiting time for each episode.
            - `self._avg_queue_length_store` (list): A list to store the average queue length for each episode.

        **Outputs**:
        - The function does not return any value but updates the following attributes:
            - `self._reward_store`: Appends `self._sum_neg_reward` for the current episode.
            - `self._cumulative_wait_store`: Appends `self._sum_waiting_time` for the current episode.
            - `self._avg_queue_length_store`: Appends the average queue length for the current episode, calculated as:
                ```
                average_queue_length = self._sum_queue_length / self._max_steps
                ```

        **Purpose**:
        - To record performance metrics for each episode, which can later be used to plot graphs or analyze the simulation's effectiveness.
        - Tracks metrics like penalties, total waiting time, and queue lengths to monitor improvements across episodes.

        **Example**:
        Suppose the episode has the following statistics:
        - `self._sum_neg_reward = -50.0` (penalty collected during the episode)
        - `self._sum_waiting_time = 300.0` seconds (total car waiting time)
        - `self._sum_queue_length = 1200` (total queued cars across all steps)
        - `self._max_steps = 100` (number of simulation steps)

        After calling this function:
        - `self._reward_store` will have an additional entry: `[-50.0]`.
        - `self._cumulative_wait_store` will have an additional entry: `[300.0]`.
        - `self._avg_queue_length_store` will have an additional entry: `[12.0]` (calculated as `1200 / 100`).

        **Returns**:
        - None
        """
        self._reward_store.append(self._sum_neg_reward)  # Append cumulative negative reward for the episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # Append total waiting time for the episode
        self._avg_queue_length_store.append(
        self._sum_queue_length / self._max_steps)  # Append average queue length per step

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times:  # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _get_queue_length(self):
        """
        Retrieve the total number of vehicles with speed = 0 across all incoming lanes.

        **Inputs**:
        - This function does not take any explicit arguments but relies on the SUMO `traci.edge` module to access real-time traffic data.

        **Outputs**:
        - `queue_length` (int): The total number of vehicles that are currently stopped (speed = 0) across all specified incoming lanes.

        **Details**:
        - For each incoming lane (north, south, east, west), the function retrieves the count of halted vehicles using `traci.edge.getLastStepHaltingNumber()`.
          - `traci.edge.getLastStepHaltingNumber(edge_id)`:
            - Returns the number of vehicles with speed = 0 in the specified edge.
            - `edge_id` is the ID of the edge (road section) connected to the traffic light.

        **Example**:
        Suppose the following vehicle counts:
        - `"N2TL"`: 5 vehicles halted.
        - `"S2TL"`: 3 vehicles halted.
        - `"E2TL"`: 7 vehicles halted.
        - `"W2TL"`: 2 vehicles halted.

        The function will compute:
        ```python
        queue_length = 5 + 3 + 7 + 2  # Total = 17
        ```
        It will return `queue_length = 17`.

        **Returns**:
        - An integer representing the total number of stopped vehicles in the incoming lanes.
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")  # Halted vehicles on the north lane
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")  # Halted vehicles on the south lane
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")  # Halted vehicles on the east lane
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")  # Halted vehicles on the west lane

        queue_length = halt_N + halt_S + halt_E + halt_W  # Total queue length
        return queue_length

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

class Testing_Simulation:
    def __init__(self, config, sumo_cmd, agent):
        # self._Memory = memory
        self._Agent =  agent
        self._TrafficGen = TrafficGenerator(config['max_steps'],config['n_cars_generated'])
        # self.epsilon = config['epsilon']
        # self._gamma = config['gamma']
        self._max_steps = config['max_steps']
        self._green_duration = config['green_duration']
        self._yellow_duration = config['yellow_duration']
        self._num_states = config['num_states']
        self._num_actions = config['num_actions']
        # self._training_epochs = config['training_epochs']
        # self._batch_size = config['batch_size']
        self._step = 0
        self._sumo_cmd = sumo_cmd
        # self._reward_store = []
        # self._cumulative_wait_store = []
        # self._avg_queue_length_store = []
        # self._epsilon = config['epsilon']
        self._reward_episode = []
        self._queue_length_episode = []

    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        old_total_wait = 0
        old_action = -1  # dummy init

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            old_total_wait = current_total_wait

            self._reward_episode.append(reward)

        # print("Total reward:", np.sum(self._reward_episode))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time

    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        if (
                self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1  # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._queue_length_episode.append(queue_length)

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times:  # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        # return np.argmax(self._Model.predict_one(state))
        return self._Agent.select_action(state, True)

    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1  # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
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
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

            # distance in meters from the traffic light -> mapping into cells
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

            # finding the lane where the car is located
            # x2TL_3 are the "turn left only" lanes
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(
                    lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[
                    car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        return state

    @property
    def queue_length_episode(self):
        return self._queue_length_episode

    @property
    def reward_episode(self):
        return self._reward_episode
