import os
import matplotlib.pyplot as plt
import numpy as np

from clt_core.core import analyze_data
import clt_core.util.info as info


def analyze_actions(env_dir, bins=5, tmax=15):
    import pickle
    import matplotlib.font_manager as font_manager

    plt.rcParams["font.family"] = "Times New Roman"

    dirs = os.listdir(env_dir)
    
    for dir in dirs:
        eval_data = pickle.load(open(os.path.join(env_dir, dir, 'eval_data'), 'rb'))

        n_episodes = len(eval_data)
        print(n_episodes)

        terminals = []
        actions = []
        successes_series = []
        for episode in eval_data:
            terminal_id = episode[-1]['terminal_class']
            if terminal_id == 2:
                actions.append(len(episode))
            terminals.append(terminal_id)
            successes_series.append(terminals.count(2) / len(terminals) * 100)

        print(actions)
        print(dir, len(actions), len(terminals), len(successes_series))
        step = int(tmax / bins)
        actions_before_singulation = []
        discrete_actions = np.arange(step, tmax + step, step)
        for na in range(step, tmax + step, step):
            print(len(np.argwhere(np.array(actions) <= na))/n_episodes)
            actions_before_singulation.append(len(np.argwhere(np.array(actions) <= na))/n_episodes)
        plt.plot(discrete_actions, actions_before_singulation, '-x', label=dir)

    font = font_manager.FontProperties(size=12, weight='bold')

    leg = plt.legend(loc='upper left', prop=font, frameon=False)

    plt.show()


def analyze_train_results(dir, smooth=0.999):
    analyze_data(dir, 'train_data', smooth)


def analyze_eval_results(exps):
    """
    Generates a summary of the results for different experiments.
    """
    import pickle
    import pandas as pd
    from tabulate import tabulate
    import numpy as np

    nr_rotations = 8

    for key in exps:
        filename = os.path.join(exps[key]['path'], 'eval_data')
        if not os.path.exists(filename):
            info.warn("File", "\"" + filename + "\", does not exist!")
            break
        with open(filename, 'rb') as outfile:
            data = pickle.load(outfile)

        n_episodes = len(data)
        exps[key]['Episodes'] = n_episodes

        terminals = []
        actions = []
        q_values = []
        rewards = []
        timesteps = 0

        primitive = []
        successes_series = []
        terminals2 = []
        for episode in data:
            terminal_id = episode[-1]['terminal_class']
            if terminal_id == 2:
                actions.append(len(episode))
                if len(episode) <=5:
                    terminals2.append(2)
            terminals.append(terminal_id)
            successes_series.append(terminals.count(2)/ len(terminals) * 100)
            for timestep in episode:
                q_values.append(timestep['q_value'])
                rewards.append(timestep['reward'])
                timesteps += 1
                primitive.append(int(np.floor(timestep['action'][0])))

        n_primitives = np.max(primitive) + 1

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # fig.suptitle('Terminal IDs distribution', fontsize=16)
        # labels, values = [], []
        # # Start from 1, because 0 means no terminal
        # for i in range(np.min(terminals), np.max(terminals) + 1):
        #     labels.append(str(i))
        #     values.append(terminals.count(i))
        # ax.bar(labels, values)
        # plt.show()

        exps[key]['Total Timesteps'] = timesteps
        exps[key]['Success Rate (%)'] = terminals.count(2) / n_episodes * 100
        exps[key]['Success Rate under 5 (%)'] = len(terminals2) / n_episodes * 100
        exps[key]['Fallen Rate (%)'] = terminals.count(3) / n_episodes * 100
        exps[key]['Collisions (%)'] = terminals.count(-11) / n_episodes * 100
        exps[key]['Max Timesteps Reached Rate (%)'] = terminals.count(1) / n_episodes * 100
        exps[key]['Num. Actions (mean)'] = np.mean(actions)
        exps[key]['Num. Actions (min)'] = np.min(actions)
        exps[key]['Num. Actions (max)'] = np.max(actions)
        exps[key]['Num. Actions (std)'] = np.std(actions)
        exps[key]['Reward (mean)'] = np.mean(rewards)
        exps[key]['Reward (min)'] = np.min(rewards)
        exps[key]['Reward (max)'] = np.max(rewards)
        exps[key]['Reward (std)'] = np.std(rewards)
        exps[key]['Q-value (mean)'] = np.mean(q_values)
        exps[key]['Q-value (min)'] = np.min(q_values)
        exps[key]['Q-value (max)'] = np.max(q_values)
        exps[key]['Q-value (std)'] = np.std(q_values)

        fig, ax = plt.subplots(2, 2)
        arr = ax[0, 0].hist(actions, bins=range(min(actions), max(actions) + 1, 1), density=True, cumulative=False)
        for i in range(len(arr[0])):
            txt = '%.2f' % arr[0][i]
            ax[0, 0].text(arr[1][i], arr[0][i], txt)
        ax[0, 0].set_title('Number of actions for singulations')
        arr = ax[0, 1].hist(actions, bins=range(min(actions), max(actions) + 1, 1), density=True, cumulative=True)
        for i in range(len(arr[0])):
            txt = '%.2f' % (arr[0][i] * terminals.count(2) / n_episodes * 100)
            ax[0, 1].text(arr[1][i], arr[0][i], txt)
        ax[0, 1].set_title('Number of actions for singulations (Cumulative)')
        ax[1, 0].plot(successes_series)
        ax[1, 0].set_title('Success rate through the 1000')

        # Plot bar plot for terminals
        labels = []
        for x in terminals:
            if x not in labels:
                labels.append(x)
        values = []
        for label in labels:
            values.append(terminals.count(label))
        # Start from 1, because 0 means no terminal
        ax[1, 1].bar(labels, values)
        ax[1, 1].set_title('Terminal IDs distribution in Evaluation')
        plt.show()

        for i in range(n_primitives):
            exps[key]['Primitive ' + str(i) + ' usage rate (%):'] = primitive.count(i) / len(primitive) * 100

    df = pd.DataFrame(exps).transpose()
    print(tabulate(df.T, headers="keys", floatfmt=".3f"))





