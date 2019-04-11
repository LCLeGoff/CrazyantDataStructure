import numpy as np
import pandas as pd

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_frame_name
from Tools.MiscellaneousTools.ArrayManipulation import get_entropy
from Tools.Plotter.Plotter import Plotter


class AnalyseFoodVelocityEntropy(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'FoodVelocityEntropy'

    def compute_w30s_entropy_mm1s_food_velocity_phi_indiv_evol(self, redo=False, redo_indiv_plot=False):
        mm = 1
        w = 30

        result_name = 'w'+str(w)+'s_entropy_mm1s_food_velocity_phi_indiv_evol'

        time_intervals = np.arange(0, 10*60)
        dtheta = np.pi / 12.
        bins = np.around(np.arange(-np.pi + dtheta/2., np.pi + dtheta, dtheta), 3)

        self.__compute_food_velocity_entropy(
            w, bins, self.category, mm, redo, redo_indiv_plot, result_name, time_intervals)

    def compute_w10s_entropy_mm1s_food_velocity_phi_indiv_evol(self, redo=False, redo_indiv_plot=False):
        mm = 1
        w = 10

        result_name = 'w'+str(w)+'s_entropy_mm1s_food_velocity_phi_indiv_evol'

        time_intervals = np.arange(0, 10*60, 1)
        dtheta = np.pi / 12.
        bins = np.around(np.arange(-np.pi + dtheta/2., np.pi + dtheta, dtheta), 3)

        self.__compute_food_velocity_entropy(
            w, bins, self.category, mm, redo, redo_indiv_plot, result_name, time_intervals)

    def compute_w1s_entropy_mm1s_food_velocity_phi_indiv_evol(self, redo=False, redo_indiv_plot=False):
        mm = 1
        w = 1

        result_name = 'w'+str(w)+'s_entropy_mm1s_food_velocity_phi_indiv_evol'

        time_intervals = np.arange(0, 10*60, 1)
        dtheta = np.pi / 12.
        bins = np.around(np.arange(-np.pi + dtheta/2., np.pi + dtheta, dtheta), 3)

        self.__compute_food_velocity_entropy(
            w, bins, self.category, mm, redo, redo_indiv_plot, result_name, time_intervals)

    def __compute_food_velocity_entropy(self,
                                        w, bins, category, mm, redo, redo_indiv_plot, result_name, time_intervals):
        if redo:
            vel_phi_name = 'mm' + str(mm) + 's_food_velocity_phi'
            self.exp.load([vel_phi_name, 'fps'])
            self.exp.add_new_empty_dataset(name=result_name, index_names='time',
                                           column_names=self.exp.id_exp_list, index_values=time_intervals,
                                           category=category, label='Evolution of the entropy of the food velocity phi',
                                           description='Time evolution of the entropy of the distribution'
                                                       ' of the angular coordinate'
                                                       ' of food velocity for each experiment')

            def compute_entropy4each_group(df: pd.DataFrame):
                exp = df.index.get_level_values(id_exp_name)[0]
                print(exp)
                fps0 = self.exp.get_value('fps', exp)
                frame0 = df.index.get_level_values(id_frame_name).min()

                w0 = w * fps0
                for time in time_intervals:
                    f0 = time * fps0 - w0 + frame0
                    f1 = time * fps0 + w0 + frame0

                    vel = df.loc[pd.IndexSlice[exp, f0:f1], :]
                    hist = np.histogram(vel, bins, normed=False)
                    hist = hist[0] / np.sum(hist[0])
                    if len(vel) != 0:
                        entropy = np.around(get_entropy(hist), 3)
                        self.exp.change_value(result_name, (time, exp), entropy)

            self.exp.groupby(vel_phi_name, id_exp_name, compute_entropy4each_group)

            self.exp.write(result_name)
        else:
            self.exp.load(result_name)
        if redo or redo_indiv_plot:

            attachment_name = 'outside_ant_carrying_intervals'
            self.exp.load(['fps', attachment_name])

            for id_exp in self.exp.get_df(result_name).columns:
                id_exp = int(id_exp)
                fps = self.exp.get_value('fps', id_exp)

                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name), column_name=id_exp)
                fig, ax = plotter.plot(xlabel='Time (s)', ylabel='Entropy',
                                       title_prefix='Exp ' + str(id_exp) + ': ')

                attachments = self.exp.get_df(attachment_name).loc[id_exp, :]
                attachments.reset_index(inplace=True)
                attachments = np.array(attachments)

                colors = plotter.color_object.create_cmap('hot_r', set(list(attachments[:, 0])))
                for id_ant, frame, inter in attachments:
                    ax.axvline(frame / fps, c=colors[str(id_ant)], alpha=0.5)
                ax.grid()
                ax.set_ylim((0.5, 4))
                plotter.save(fig, name=id_exp, sub_folder=result_name)
