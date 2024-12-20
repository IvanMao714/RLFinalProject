import datetime
import os
from shutil import copyfile

from RLfinal.agent.QlearningAgent import QLearningAgent
from RLfinal.agent.SACAgent import SACAgent

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Temporary workaround to handle library conflicts

# from numpy.distutils.command.config import config

from RLfinal.agent.DQNAgent import DQNAgent
from RLfinal.config.utils import import_configuration
from RLfinal.memory import Memory
from RLfinal.plot.visualization import save_data_and_plot
from RLfinal.simulation import Training_Simulation
from RLfinal.utils import set_sumo, set_train_path




class Trainer:

    def __init__(self, model_name):
        # if type == "train":
        #     self.config = import_train_configuration(config_file='training_ac_settings.ini')
        #     self.path = set_train_path(self.config['models_path_name'])
        # else:
        #     self.config = import_test_configuration(config_file='testing_ac_settings.ini')
        self.config = import_configuration(model_name, 'train')
        self.sumo_cmd = set_sumo(self.config['gui'], self.config['sumocfg_file_name'], self.config['max_steps'])
        self.path = set_train_path(self.config['models_path_name'], model_name)
        self.model_name = model_name
        self.memory = Memory(self.config)

        # Initialize the agent based on the specified model name
        if model_name == "DQN" or model_name == "DDQN" or model_name == "DDDQN":
            self.agent = DQNAgent(self.config, self.memory)
        elif model_name == "SAC":
            self.agent = SACAgent(self.config, self.memory)
        elif model_name == "Q-learning":
            self.agent = QLearningAgent(self.config)
        else:
            raise ValueError("Unknown model name: {}".format(model_name))

        # Initialize the training simulation
        self.simulation = Training_Simulation(self.config, self.sumo_cmd, self.agent, self.memory, model_name)


    def train(self):

        """
        Train the model for a specified number of episodes.
        """
        episode = 0
        timestamp_start = datetime.datetime.now() # Record the start time

        # Main training loop
        while episode < self.config['total_episodes']:
            print('\n----- Episode', str(episode + 1), 'of', str(self.config['total_episodes']))
            # epsilon = 1.0 - (episode / self.config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
            simulation_time, training_time = self.simulation.run(episode)  # run the simulation for the current episode
            print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:',
                  round(simulation_time + training_time, 1), 's')
            episode += 1
            if episode % 20 == 0:
                self.agent.save(self.model_name, episode, self.path)
                # print(self.simulation.reward_store)
        # Save the final model after all episodes are complete
        self.agent.save(self.model_name, episode, self.path)

        # Generate and save performance plots
        save_data_and_plot(data=self.simulation.reward_store, filename='reward', train_type='train',xlabel='Episode', ylabel='Cumulative negative reward', path=self.path,dpi=96)
        save_data_and_plot(data=self.simulation.cumulative_wait_store, filename='delay', train_type='train', xlabel='Episode',
                                         ylabel='Cumulative delay (s)', path=self.path,dpi=96)
        save_data_and_plot(data=self.simulation.avg_queue_length_store, filename='queue', train_type='train', xlabel='Episode',
                                         ylabel='Average queue length (vehicles)', path=self.path,dpi=96)

        # Log session details
        print("\n----- Start time:", timestamp_start)
        print("----- End time:", datetime.datetime.now())
        print("----- Session info saved at:", self.path)



        # copyfile(src='training_settings.ini', dst=os.path.join(self.path, 'training_settings.ini'))



if __name__ == '__main__':
    # Uncomment the desired model to train
    
    # Trainer('DQN').train()
    Trainer('DDQN').train()
    # Trainer("Q-learning").train()
