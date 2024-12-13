# import datetime
# import os
# from shutil import copyfile
#
# from .agent.SACAgent import SACAgent
#
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Temporary workaround
#
# # from numpy.distutils.command.config import config
#
# from .agent.DQNAgent import DQNAgent
# from .config.utils import import_configuration
# from .memory import Memory
# from .plot.visualization import save_data_and_plot
# from .simulation import Training_Simulation
# from .utils import set_sumo, set_train_path
#
#
#
#
# class Trainer:
#
#     def __init__(self, model_name):
#         # if type == "train":
#         #     self.config = import_train_configuration(config_file='training_ac_settings.ini')
#         #     self.path = set_train_path(self.config['models_path_name'])
#         # else:
#         #     self.config = import_test_configuration(config_file='testing_ac_settings.ini')
#         self.config = import_configuration(model_name,'train')
#         self.sumo_cmd = set_sumo(self.config['gui'], self.config['sumocfg_file_name'], self.config['max_steps'])
#         self.path = set_train_path(self.config['models_path_name'], model_name)
#         self.model_name = model_name
#         self.memory = Memory(self.config)
#         if model_name == "DQN" or model_name == "DDQN" or model_name == "DDDQN":
#             self.agent = DQNAgent(self.config, self.memory)
#         elif model_name == "SAC":
#             self.agent = SACAgent(self.config, self.memory)
#
#         self.simulation = Training_Simulation(self.config, self.sumo_cmd, self.agent, self.memory)
#
#
#     def train(self):
#         episode = 0
#         timestamp_start = datetime.datetime.now()
#
#         while episode < self.config['total_episodes']:
#             print('\n----- Episode', str(episode + 1), 'of', str(self.config['total_episodes']))
#             # epsilon = 1.0 - (episode / self.config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
#             simulation_time, training_time = self.simulation.run(episode)  # run the simulation
#             print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:',
#                   round(simulation_time + training_time, 1), 's')
#             episode += 1
#             if episode % 20 == 0:
#                 self.agent.save(self.model_name, episode, self.path)
#                 # print(self.simulation.reward_store)
#         self.agent.save(self.model_name, episode, self.path)
#         save_data_and_plot(data=self.simulation.reward_store, filename='reward', train_type='train',xlabel='Episode', ylabel='Cumulative negative reward', path=self.path,dpi=96)
#         save_data_and_plot(data=self.simulation.cumulative_wait_store, filename='delay', train_type='train', xlabel='Episode',
#                                          ylabel='Cumulative delay (s)', path=self.path,dpi=96)
#         save_data_and_plot(data=self.simulation.avg_queue_length_store, filename='queue', train_type='train', xlabel='Episode',
#                                          ylabel='Average queue length (vehicles)', path=self.path,dpi=96)
#         print("\n----- Start time:", timestamp_start)
#         print("----- End time:", datetime.datetime.now())
#         print("----- Session info saved at:", self.path)
#
#
#
#         # copyfile(src='training_settings.ini', dst=os.path.join(self.path, 'training_settings.ini'))
#
#
#
# if __name__ == '__main__':
#     Trainer("DDQN").train()





import datetime
import os
from shutil import copyfile

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Temporary workaround

from agent.SACAgent import SACAgent
from agent.DQNAgent import DQNAgent
from agent.QlearningAgent import QLearningAgent  # 引入修改后的QLearningAgent
from config.utils import import_configuration
from memory import Memory
from plot.visualization import save_data_and_plot
from simulation import Training_Simulation
from utils import set_sumo, set_train_path

class Trainer:

    def __init__(self, model_name):
        self.config = import_configuration(model_name, 'train')
        self.sumo_cmd = set_sumo(self.config['gui'], self.config['sumocfg_file_name'], self.config['max_steps'])
        self.path = set_train_path(self.config['models_path_name'], model_name)
        self.model_name = model_name
        self.memory = Memory(self.config)

        # 根据模型名称选择相应的智能体
        if model_name == "DQN" or model_name == "DDQN" or model_name == "DDDQN":
            self.agent = DQNAgent(self.config, self.memory)
        elif model_name == "SAC":
            self.agent = SACAgent(self.config, self.memory)
        elif model_name == "Q-learning":
            # 直接使用config创建QLearningAgent
            self.agent = QLearningAgent(self.config)
        else:
            raise ValueError("Unknown model name: {}".format(model_name))

        self.simulation = Training_Simulation(self.config, self.sumo_cmd, self.agent, self.memory)
        # 请确保在 Training_Simulation 中，对于 Q-learning 是在每步环境交互后就直接调用 self.agent.train(...)
        # 而不是像 DQN 那样先存储在 memory 再批量取样训练。

    def train(self):
        episode = 0
        timestamp_start = datetime.datetime.now()

        while episode < self.config['total_episodes']:
            print('\n----- Episode', str(episode + 1), 'of', str(self.config['total_episodes']))
            simulation_time, training_time = self.simulation.run(episode)  # run the simulation
            print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:',
                  round(simulation_time + training_time, 1), 's')
            episode += 1

            # 定期保存模型
            if episode % 20 == 0:
                self.agent.save(self.model_name, episode, self.path)

        # 最后保存一次模型
        self.agent.save(self.model_name, episode, self.path)

        # 保存并绘制训练数据
        save_data_and_plot(data=self.simulation.reward_store,
                           filename='reward',
                           train_type='train',
                           xlabel='Episode',
                           ylabel='Cumulative negative reward',
                           path=self.path,
                           dpi=96)

        save_data_and_plot(data=self.simulation.cumulative_wait_store,
                           filename='delay',
                           train_type='train',
                           xlabel='Episode',
                           ylabel='Cumulative delay (s)',
                           path=self.path,
                           dpi=96)

        save_data_and_plot(data=self.simulation.avg_queue_length_store,
                           filename='queue',
                           train_type='train',
                           xlabel='Episode',
                           ylabel='Average queue length (vehicles)',
                           path=self.path,
                           dpi=96)

        print("\n----- Start time:", timestamp_start)
        print("----- End time:", datetime.datetime.now())
        print("----- Session info saved at:", self.path)


if __name__ == '__main__':
    Trainer("Q-learning").train()
