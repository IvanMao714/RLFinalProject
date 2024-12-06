import ast
import configparser
import os


def import_configuration(model, type):
    """
    Read the config file regarding the training and import its content
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the config file
    simulation_config_path = os.path.join(base_dir, 'simulation_settings.ini')
    base_config_path = os.path.join(base_dir, 'base_settings.ini')
    training_config_path = os.path.join(base_dir, 'training_settings.ini')

    config = {}
    content = configparser.ConfigParser()
    content.read(simulation_config_path)

    config['gui'] = content['simulation'].getboolean('gui')
    config['total_episodes'] = content['simulation'].getint('total_episodes')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')

    content.read(base_config_path)
    config['device'] = content['device'].get('device_type')
    config['models_path_name'] = content['dir']['models_path_name']
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    config['memory_size_max'] = content['memory'].getint('memory_size_max')
    config['num_states'] = content['agent'].getint('num_states')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['gamma'] = content['agent'].getfloat('gamma')
    config['batch_size'] = content['agent'].getint('batch_size')
    config['epsilon'] = content['agent'].getfloat('epsilon')


    if type == 'train':
        content.read(training_config_path)
        if model == 'DQN':
            config['learning_rate'] = content['dqn_model'].getfloat('learning_rate')
            config['training_epochs'] = content['dqn_model'].getint('training_epochs')
            config['use_double_dqn'] = content['dqn_model'].getboolean('use_double_dqn')
            config['use_dueling_network'] = content['dqn_model'].getboolean('use_dueling_network')
            config['hidden_layers'] = ast.literal_eval(content['dqn_model']['hidden_layers'])
        if model == 'DDQN':
            config['learning_rate'] = content['ddqn_model'].getfloat('learning_rate')
            config['training_epochs'] = content['ddqn_model'].getint('training_epochs')
            config['use_double_dqn'] = content['ddqn_model'].getboolean('use_double_dqn')
            config['use_dueling_network'] = content['ddqn_model'].getboolean('use_dueling_network')
            config['hidden_layers'] = ast.literal_eval(content['ddqn_model']['hidden_layers'])
        if model == 'DDDQN':
            config['learning_rate'] = content['dddqn_model'].getfloat('learning_rate')
            config['training_epochs'] = content['dddqn_model'].getint('training_epochs')
            config['use_double_dqn'] = content['dddqn_model'].getboolean('use_double_dqn')
            config['use_dueling_network'] = content['dddqn_model'].getboolean('use_dueling_network')
            config['hidden_layers'] = ast.literal_eval(content['dddqn_model']['hidden_layers'])
        if model == 'SAC':
            config['learning_rate'] = content['sac_model'].getfloat('learning_rate')
            config['training_epochs'] = content['sac_model'].getint('training_epochs')
            config['hidden_layers'] = ast.literal_eval(content['sac_model']['hidden_layers'])
            config['adaptive_alpha'] = content['sac_model'].getboolean('adaptive_alpha')
            config['alpha'] = content['sac_model'].getfloat('alpha')

    # print(config)
    return config


# def import_test_configuration(config_file):
#     """
#     Read the config file regarding the testing and import its content
#     """
#     content = configparser.ConfigParser()
#     content.read(config_file)
#     config = {}
#     config['gui'] = content['simulation'].getboolean('gui')
#     config['max_steps'] = content['simulation'].getint('max_steps')
#     config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
#     config['episode_seed'] = content['simulation'].getint('episode_seed')
#     config['green_duration'] = content['simulation'].getint('green_duration')
#     config['yellow_duration'] = content['simulation'].getint('yellow_duration')
#     config['num_states'] = content['agent'].getint('num_states')
#     config['num_actions'] = content['agent'].getint('num_actions')
#     config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
#     config['models_path_name'] = content['dir']['models_path_name']
#     config['model_to_test'] = content['dir'].getint('model_to_test')
#     return config
#
# def import_dqn_training_configuration(config_file):
#     """
#     Read the config file regarding the training and import its content
#     """
#     content = configparser.ConfigParser()
#     content.read(config_file)
#     config = {}
#
#     config['gui'] = content['simulation'].getboolean('gui')

# if __name__ == '__main__':
    # import_train_configuration('./simulation_settings.ini')