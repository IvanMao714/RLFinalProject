import os
import timeit
from distutils.command.config import config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Temporary workaround to resolve library conflicts

from agent.DQNAgent import DQNAgent
from agent.SACAgent import SACAgent
from agent.QlearningAgent import QLearningAgent  # Import Q-learning agent
from config.utils import import_configuration
from plot.visualization import save_data_and_plot
from simulation import Testing_Simulation
from utils import set_sumo


class Tester:
    def __init__(self, model_name):


        """
        Initialize the Tester class for evaluating trained models.
        Args:
            model_name (str): The name of the model to be tested (e.g., 'DQN', 'SAC', 'Q-learning').
        """
        # Load configuration for testing based on the selected model
        self.config = import_configuration(model_name, 'test')
        print(self.config['gui'])

        # Set up SUMO simulation command using the specified configuration
        self.sumo_cmd = set_sumo(self.config['gui'], self.config['sumocfg_file_name'], self.config['max_steps'])

        # Store the model name and initialize memory (if required)
        self.model_name = model_name
        self.memory = None


        # Select the appropriate agent and model path based on the model name
        if model_name in ["DQN", "DDQN", "DDDQN"]:

            # Define the path to the saved model for DQN-related agents
            self.model_path = os.path.join(os.getcwd(), 'models', model_name, model_name + "_1")
            self.agent = DQNAgent(self.config, self.memory)
            # Load the trained model; assumes step=50 (adjust based on training save step)
            self.agent.load(model_name, 50, self.model_path)


        elif model_name == "SAC":

            # Define the path to the saved model for SAC
            self.model_path = os.path.join(os.getcwd(), 'models', model_name, model_name + "_1")
            self.agent = SACAgent(self.config, self.memory)

            # Load the trained model; assumes step=50
            self.agent.load(model_name, 50, self.model_path)

        elif model_name == "Q-learning":
            # Define the path to the saved Q-learning model
            self.model_path = os.path.join(os.getcwd(), 'models', model_name, model_name + "_9")
            self.agent = QLearningAgent(self.config)
            self.agent.load(model_name, 50, self.model_path)  # Load the saved Q-table; assumes step=50

        else:
            raise ValueError("Unknown model name: {}".format(model_name))


        # Initialize the testing simulation with the chosen agent
        self.simulation = Testing_Simulation(self.config, self.sumo_cmd, self.agent)

    def run(self):

        """
        Run the testing simulation for the specified model.
        """
        print('\n----- Test episode')
        simulation_time = self.simulation.run(self.config['episode_seed'])  # Run the simulation and measure the simulation time
        print('Simulation time:', simulation_time, 's')


        # Save and plot the test results for rewards
        save_data_and_plot(data=self.simulation.reward_episode,
                           filename='reward',
                           train_type='test',
                           xlabel='Episode',
                           ylabel='Cumulative negative reward',
                           path=self.model_path,
                           dpi=96)
        
        # Save and plot the test results for queue lengths
        save_data_and_plot(data=self.simulation.queue_length_episode,
                           filename='queue',
                           train_type='test',
                           xlabel='Step',
                           ylabel='Queue Length (vehicles)',
                           path=self.model_path,
                           dpi=96)


if __name__ == '__main__':
    # Specify the model to test, e.g., 'Q-learning'
    test = Tester('Q-learning')
    test.run()

