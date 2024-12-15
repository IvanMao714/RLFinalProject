import matplotlib.pyplot as plt
import os

import pandas as pd


def save_data_and_plot(data, filename,train_type, xlabel, ylabel, path, dpi):
    """
    Produce a plot of performance of the agent over the session and save the relative data to txt
    """
    min_val = min(data)
    max_val = max(data)

    plt.rcParams.update({'font.size': 24})  # set bigger font size

    plt.plot(data)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.margins(0)
    plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(os.path.join(path, 'plot_'+train_type+"_"+filename+'.png'), dpi=dpi)
    plt.close("all")

    with open(os.path.join(path, 'plot_'+train_type+"_"+filename + '_data.txt'), "w") as file:
        for value in data:
                file.write("%s\n" % value)


def mode_performance_comparison():
    # Directories and file names
    # file_names = ['plot_delay_data', 'plot_queue_data', ]
    file_names = ['plot_delay_data', 'plot_reward_data', 'plot_queue_data']
    model_paths = ['../models/DQN/DQN_1', '../models/DDQN/DDQN_1', '../models/DDDQN/DDDQN_1', '../models/SAC/SAC_1', "../models/Q-learning/Q-learning_1"]

    # Initialize a dictionary to store DataFrames
    data_frames = {}

    # Read data into pandas DataFrames
    for file in file_names:
        data_dict = {}
        for model_path in model_paths:
            print(f'{file}: {model_path}')
            file_path = os.path.join(model_path, file + ".txt")
            model_name = os.path.basename(model_path)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data_dict[model_name] = [float(line.strip()) for line in f]
            else:
                data_dict[model_name] = []  # Handle missing files with empty data
            print(len(data_dict[model_name]))
        data_frames[file] = pd.DataFrame(data_dict)
    # print(data_frames)
    for file, df in data_frames.items():
        plt.figure(figsize=(10, 6))
        for column in df.columns:
            plt.plot(df.index, df[column], linestyle='-', label=column)
        plt.title(f"{file.replace('_', ' ').title()} Comparison")
        plt.xlabel("Episode")
        plt.ylabel(file.split('_')[1].title())
        plt.grid(True)
        plt.legend()
        plt.savefig(f'./Comparison-{file}.png')
        plt.show()

if __name__ == '__main__':
    mode_performance_comparison()