import matplotlib.pyplot as plt
import os




def save_data_and_plot(data, filename, xlabel, ylabel, path, dpi):
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
    fig.savefig(os.path.join(path, 'plot_'+filename+'.png'), dpi=dpi)
    plt.close("all")

    with open(os.path.join(path, 'plot_'+filename + '_data.txt'), "w") as file:
        for value in data:
                file.write("%s\n" % value)