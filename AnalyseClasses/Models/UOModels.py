import numpy as np
import random as rd

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name
from ExperimentGroups import ExperimentGroups
from Tools.Plotter.Plotter import Plotter


class BaseModels:
    def __init__(self, parameter_names):
        self.parameter_names = parameter_names
        self.para = ModelParameters(self.parameter_names)

        self.t = None
        self.last_attachment_time = None
        self.id_exp = None
        self.orientation = None
        self.name_column = None

        self.time_name = 't'

    def get_p_attachment(self):
        if self.para.p_attachment == 'power':
            p_attachment = 1 / ((self.t - self.last_attachment_time) + 1)
        else:
            p_attachment = self.para.p_attachment
        return p_attachment


class ModelParameters:
    def __init__(self, parameter_name_list):
        self.parameter_name_list = parameter_name_list

        for parameter_name in parameter_name_list:
            self.__dict__[parameter_name] = None

    def change_parameter_values(self, value_dict: dict):
        for parameter_name in self.parameter_name_list:
            self.__dict__[parameter_name] = value_dict[parameter_name]

    def get_parameter_tuple(self):
        res = []
        for parameter_name in self.parameter_name_list:
            res.append(self.__dict__[parameter_name])

        return tuple(res)


class UOSimpleModel(BaseModels):
    def __init__(self, root, group, duration=300, n_replica=500, new=False):

        self.exp = ExperimentGroups(root, group)
        self.name = 'UOSimpleModel'
        parameter_names = ['c', 'p_attachment', 'var_orientation', 'var_perception', 'var_information']

        BaseModels.__init__(self, parameter_names)

        self.duration = duration
        self.n_replica = n_replica

        self.res = None

        self.init(new)

    def init(self, new):
        if new is True or not self.exp.is_name_existing(self.name):
            index = [(id_exp, t) for id_exp in range(1, self.n_replica+1) for t in range(self.duration+1)]

            self.exp.add_new_empty_dataset(name=self.name, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO simple model',
                                           description='UO simple model with parameter c=%, '
                                                       'attachment probability=p+attachment,'
                                                       'orientation variance = var_orientation, '
                                                       'orientation perception = var_perception, '
                                                       'orientation information = var_information')
        else:

            self.exp.load(self.name)
            self.n_replica = max(self.exp.get_index(self.name).get_level_values(id_exp_name))
            self.duration = max(self.exp.get_index(self.name).get_level_values(self.time_name))

    def run(self, para_value):

        self.para.change_parameter_values(para_value)
        self.name_column = str(self.para.get_parameter_tuple())

        self.res = []
        print(self.name_column)

        for id_exp in range(1, self.n_replica + 1):
            self.id_exp = id_exp
            self.orientation = rd.uniform(-np.pi, np.pi)
            self.res.append(self.orientation)
            self.t = 0
            self.last_attachment_time = 0

            for t in range(1, self.duration + 1):
                self.step()

        self.exp.get_df(self.name)[self.name_column] = self.res

    def step(self):
        self.t += 1

        u = rd.random()
        p_attachment = self.get_p_attachment()

        if u < p_attachment:
            eta = rd.normalvariate(0, self.para.var_perception)
            theta_ant = rd.normalvariate(0, self.para.var_information)

            dtheta = theta_ant-self.orientation+eta
            self.orientation += self.para.c*dtheta

            self.last_attachment_time = self.t

        rho = rd.normalvariate(0, self.para.var_orientation)
        self.orientation += rho

        self.res.append(np.around(self.orientation, 3))

    def write(self):
        self.exp.write(self.name)


class UOConfidenceModel(BaseModels):
    def __init__(self, root, group, duration=300, n_replica=500, new=False):

        self.exp = ExperimentGroups(root, group)
        self.name = 'UOConfidenceModel'
        self.name_confidence = 'UOConfidenceModel_confidence'

        parameter_names = ['p_attachment', 'var_orientation', 'var_perception', 'var_information']
        BaseModels.__init__(self, parameter_names)

        self.duration = duration
        self.n_replica = n_replica

        self.init(new)

        self.res_orientation = None
        self.res_confidence = None
        self.c_ant = None
        self.confidence = None

    def init(self, new):
        if new is True or not self.exp.is_name_existing(self.name):
            index = [(id_exp, t) for id_exp in range(1, self.n_replica+1) for t in range(self.duration+1)]

            self.exp.add_new_empty_dataset(name=self.name, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO confidence model',
                                           description='UO confidence model with parameter '
                                                       'attachment probability=p+attachment,'
                                                       'orientation variance = var_orientation, '
                                                       'orientation perception = var_perception, '
                                                       'orientation information = var_information')

            self.exp.add_new_empty_dataset(name=self.name_confidence, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='Confidence of the UO confidence model',
                                           description='Confidence of the UO confidence model with parameter '
                                                       'attachment probability=p+attachment,'
                                                       'orientation variance = var_orientation, '
                                                       'orientation perception = var_perception, '
                                                       'orientation information = var_information')
        else:

            self.exp.load([self.name, self.name_confidence])
            self.n_replica = max(self.exp.get_index(self.name).get_level_values(id_exp_name))
            self.duration = max(self.exp.get_index(self.name).get_level_values(self.time_name))

    def run(self, para_value):

        self.para.change_parameter_values(para_value)
        self.name_column = str(self.para.get_parameter_tuple())

        self.c_ant = 1/self.para.var_information
        self.name_column = str(
            (self.para.p_attachment, self.para.var_orientation, self.para.var_perception, self.para.var_information))

        self.res_orientation = []
        self.res_confidence = []
        print(self.name_column)

        for id_exp in range(1, self.n_replica + 1):
            self.id_exp = id_exp
            self.orientation = rd.uniform(-np.pi, np.pi)
            self.confidence = 1/self.para.var_orientation

            self.res_orientation.append(self.orientation)
            self.res_confidence.append(self.confidence)

            self.t = 0
            for t in range(1, self.duration + 1):
                self.step()

        self.exp.get_df(self.name)[self.name_column] = self.res_orientation
        self.exp.get_df(self.name_confidence)[self.name_column] = self.res_confidence

    def step(self):
        self.t += 1

        u = rd.random()
        p_attachment = self.get_p_attachment()

        if u < p_attachment:
            eta = rd.normalvariate(0, self.para.var_perception)
            theta_ant = rd.normalvariate(0, self.para.var_information)

            dtheta = theta_ant-self.orientation+eta

            self.orientation += self.c_ant*dtheta/(self.confidence+self.c_ant)
            self.confidence = 1/(self.para.var_orientation+1/(self.confidence+self.c_ant))
        else:
            self.confidence = 1/(self.para.var_orientation+1/self.confidence)

        rho = rd.normalvariate(0, self.para.var_orientation)
        self.orientation += rho

        self.res_orientation.append(np.around(self.orientation, 3))
        self.res_confidence.append(np.around(self.confidence, 3))

    def write(self):
        self.exp.write([self.name, self.name_confidence])


class AnalyseUOModel(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'Models'

    def plot_simple_model_evol(self, n=None, m=None):

        name = 'UOSimpleModel'
        self._plot_hist_evol(name, n, m, 'simple')

    def plot_confidence_model_evol(self, n=None, m=None):

        name = 'UOConfidenceModel'
        self._plot_hist_evol(name, n, m, 'confidence')
        self._plot_confidence_evol(name+'_confidence', n, m, 'confidence')

    def _plot_confidence_evol(self, name, n=None, m=None, model=None):

        self.exp.load(name)

        column_names = self.exp.get_data_object(name).get_column_names()

        if n is None:
            n = int(np.floor(np.sqrt(len(column_names))))
            m = int(np.ceil(len(column_names) / n))

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(name))
        fig, ax = plotter.create_plot(figsize=(4 * m, 4.2 * n), nrows=n, ncols=m, top=0.9, bottom=0.05)

        for k, column_name in enumerate(column_names):
            i = int(k / m)
            j = k % m

            if isinstance(ax, np.ndarray):
                if len(ax.shape) == 1:
                    ax0 = ax[k]
                else:
                    ax0 = ax[i, j]
            else:
                ax0 = ax

            list_id_exp = set(self.exp.get_df(name).index.get_level_values(id_exp_name))
            df = self.exp.get_df(name).loc[1, :]
            df2 = df[column_name]
            for id_exp in range(2, max(list_id_exp)):
                df = self.exp.get_df(name).loc[id_exp, :]
                df2 += df[column_name]

            self.exp.add_new_dataset_from_df(df=df2/float(len(list_id_exp)), name='temp',
                                             category=self.category, replace=True)

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
            plotter.plot(xlabel='orientation', ylabel='PDF', normed=True, preplot=(fig, ax0), title=column_name)
            ax0.axhline(1/float(column_name.split(',')[-1][:-1]), c='k', ls='--')

        if model == 'simple':
            fig.suptitle(r"Simple model, parameters $(c, p_{att}, \sigma_{orient}, \sigma_{perc}, \sigma_{info})$",
                         fontsize=15)
        elif model == 'confidence':
            fig.suptitle(r"Confidence model, parameters $(p_{att}, \sigma_{orient}, \sigma_{perc}, \sigma_{info})$",
                         fontsize=15)

        plotter.save(fig, name=name)

    def _plot_hist_evol(self, name, n=None, m=None, model=None):

        self.exp.load(name)
        self.exp.get_data_object(name).df = np.abs(self.exp.get_df(name))

        time_name = 't'
        column_names = self.exp.get_data_object(name).get_column_names()

        time_intervals = np.arange(0, 5., .5) * 60
        dtheta = np.pi / 25.
        bins = np.arange(0, np.pi, dtheta)

        if n is None:
            n = int(np.floor(np.sqrt(len(column_names))))
            m = int(np.ceil(len(column_names) / n))

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(name))
        fig, ax = plotter.create_plot(figsize=(4 * m, 4 * n), nrows=n, ncols=m, top=0.9, bottom=0.05)

        for k, column_name in enumerate(column_names):

            temp_name = self.exp.hist1d_evolution(name_to_hist=name, index_intervals=time_intervals, bins=bins,
                                                  index_name=time_name, column_to_hist=column_name, replace=True)

            i = int(k / m)
            j = k % m
            if isinstance(ax, np.ndarray):
                if len(ax.shape) == 1:
                    ax0 = ax[k]
                else:
                    ax0 = ax[i, j]
            else:
                ax0 = ax

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(temp_name))
            plotter.plot(xlabel='orientation', ylabel='PDF', normed=True, preplot=(fig, ax0), title=column_name)

        if model == 'simple':
            fig.suptitle(r"Simple model, parameters $(c, p_{att}, \sigma_{orient}, \sigma_{perc}, \sigma_{info})$",
                         fontsize=15)
        elif model == 'confidence':
            fig.suptitle(r"Confidence model, parameters $(p_{att}, \sigma_{orient}, \sigma_{perc}, \sigma_{info})$",
                         fontsize=15)

        plotter.save(fig)
