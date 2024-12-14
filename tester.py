
# import os
# import timeit
# from distutils.command.config import config
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Temporary workaround
# from .agent.DQNAgent import DQNAgent
# from .agent.SACAgent import SACAgent
# from .config.utils import import_configuration
# from .plot.visualization import save_data_and_plot
# from .simulation import Testing_Simulation
# from .utils import set_sumo
#
#
# class Tester:
#     def __init__(self, model_name):
#         self.config = import_configuration(model_name, 'test')
#         print(self.config['gui'])
#         self.sumo_cmd = set_sumo(self.config['gui'], self.config['sumocfg_file_name'], self.config['max_steps'])
#         self.model_name = model_name
#         self.memory = None
#
#         if model_name == "DQN" or model_name == "DDQN" or model_name == "DDDQN":
#             self.model_path = os.path.join(os.getcwd(), 'models', model_name, model_name+"_1")
#             self.agent = DQNAgent(self.config, self.memory)
#
#
#         elif model_name == "SAC":
#             self.model_path = os.path.join(os.getcwd(), 'models', model_name, model_name + "_1")
#             self.agent = SACAgent(self.config, self.memory)
#         self.agent.load(model_name, 50, self.model_path)
#         self.simulation = Testing_Simulation(self.config, self.sumo_cmd, self.agent)
#
#     def run(self):
#         print('\n----- Test episode')
#         simulation_time = self.simulation.run(self.config['episode_seed'])  # run the simulation
#         print('Simulation time:', simulation_time, 's')
#         save_data_and_plot(data=self.simulation.reward_episode, filename='reward', train_type='test',xlabel='Episode',
#                            ylabel='Cumulative negative reward', path=self.model_path, dpi=96)
#         save_data_and_plot(data=self.simulation.queue_length_episode, filename='queue', train_type='test', xlabel='Step',
#                            ylabel='Queue Length (vehicles)', path=self.model_path, dpi=96)
#
# if __name__ == '__main__':
#     test = Tester('SAC')
#     test.run()

import os
import timeit
from distutils.command.config import config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Temporary workaround

from agent.DQNAgent import DQNAgent
from agent.SACAgent import SACAgent
from agent.QlearningAgent import QLearningAgent  # 引入Q-learning智能体
from config.utils import import_configuration
from plot.visualization import save_data_and_plot
from simulation import Testing_Simulation
from utils import set_sumo


class Tester:
    def __init__(self, model_name):
        self.config = import_configuration(model_name, 'test')
        print(self.config['gui'])
        self.sumo_cmd = set_sumo(self.config['gui'], self.config['sumocfg_file_name'], self.config['max_steps'])
        self.model_name = model_name
        self.memory = None


        # 根据不同模型类型选择不同的智能体与模型路径
        if model_name in ["DQN", "DDQN", "DDDQN"]:
            self.model_path = os.path.join(os.getcwd(), 'models', model_name, model_name + "_1")
            self.agent = DQNAgent(self.config, self.memory)
            # 假设您在训练阶段最后保存时使用了step=50作为示例，如需修改请对齐训练保存的步数
            self.agent.load(model_name, 50, self.model_path)


        elif model_name == "SAC":
            self.model_path = os.path.join(os.getcwd(), 'models', model_name, model_name + "_1")
            self.agent = SACAgent(self.config, self.memory)

            self.agent.load(model_name, 50, self.model_path)

        elif model_name == "Q-learning":
            # 对于Q-learning模型，路径与DQN/SAC类似，假设在训练时也使用了类似路径和保存命名规则。
            self.model_path = os.path.join(os.getcwd(), 'models', model_name, model_name + "_9")
            self.agent = QLearningAgent(self.config)
            self.agent.load(model_name, 50, self.model_path)  # 加载Q表

        else:
            raise ValueError("Unknown model name: {}".format(model_name))


        self.simulation = Testing_Simulation(self.config, self.sumo_cmd, self.agent)

    def run(self):
        print('\n----- Test episode')
        simulation_time = self.simulation.run(self.config['episode_seed'])  # run the simulation
        print('Simulation time:', simulation_time, 's')


        # 保存并绘制测试数据
        save_data_and_plot(data=self.simulation.reward_episode,
                           filename='reward',
                           train_type='test',
                           xlabel='Episode',
                           ylabel='Cumulative negative reward',
                           path=self.model_path,
                           dpi=96)

        save_data_and_plot(data=self.simulation.queue_length_episode,
                           filename='queue',
                           train_type='test',
                           xlabel='Step',
                           ylabel='Queue Length (vehicles)',
                           path=self.model_path,
                           dpi=96)


if __name__ == '__main__':
    # 将此处修改为所需的模型名称，例如 Q-learning
    test = Tester('Q-learning')
    test.run()

