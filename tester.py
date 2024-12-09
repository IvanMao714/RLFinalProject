import os
import timeit
from distutils.command.config import config
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Temporary workaround
from RLfinal.agent.DQNAgent import DQNAgent
from RLfinal.agent.SACAgent import SACAgent
from RLfinal.config.utils import import_configuration
from RLfinal.plot.visualization import save_data_and_plot
from RLfinal.simulation import Testing_Simulation
from RLfinal.utils import set_sumo


class Tester:
    def __init__(self, model_name):
        self.config = import_configuration(model_name, 'test')
        print(self.config['gui'])
        self.sumo_cmd = set_sumo(self.config['gui'], self.config['sumocfg_file_name'], self.config['max_steps'])
        self.model_name = model_name
        self.memory = None

        if model_name == "DQN" or model_name == "DDQN" or model_name == "DDDQN":
            self.model_path = os.path.join(os.getcwd(), 'models', model_name, model_name+"_1")
            self.agent = DQNAgent(self.config, self.memory)


        elif model_name == "SAC":
            self.model_path = os.path.join(os.getcwd(), 'models', model_name, model_name + "_1")
            self.agent = SACAgent(self.config, self.memory)
        self.agent.load(model_name, 50, self.model_path)
        self.simulation = Testing_Simulation(self.config, self.sumo_cmd, self.agent)

    def run(self):
        print('\n----- Test episode')
        simulation_time = self.simulation.run(self.config['episode_seed'])  # run the simulation
        print('Simulation time:', simulation_time, 's')
        save_data_and_plot(data=self.simulation.reward_episode, filename='reward', train_type='test',xlabel='Episode',
                           ylabel='Cumulative negative reward', path=self.model_path, dpi=96)
        save_data_and_plot(data=self.simulation.queue_length_episode, filename='queue', train_type='test', xlabel='Step',
                           ylabel='Queue Length (vehicles)', path=self.model_path, dpi=96)

if __name__ == '__main__':
    test = Tester('SAC')
    test.run()