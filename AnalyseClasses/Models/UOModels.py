import numpy as np
import random as rd
import scipy.stats as scs
import pandas as pd

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name
from ExperimentGroups import ExperimentGroups
from Tools.MiscellaneousTools.ArrayManipulation import get_index_interval_containing
from Tools.MiscellaneousTools.Geometry import angle_distance
from Tools.Plotter.Plotter import Plotter


class BaseModels:
    def __init__(self, parameter_names):
        self.parameter_names = parameter_names
        self.para = ModelParameters(self.parameter_names)

        self.t = None
        self.last_attachment_time = 0
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
        parameter_names = ['c', 'p_attachment', 'var_orientation', 'var_information']

        BaseModels.__init__(self, parameter_names)

        self.duration = duration
        self.n_replica = n_replica

        self.res = None

        self.init(new)

    def init(self, new):
        if new is True or not self.exp.is_name_existing(self.name):
            index = [(id_exp, t*100) for id_exp in range(1, self.n_replica+1) for t in range(self.duration+1)]

            self.exp.add_new_empty_dataset(name=self.name, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO simple model',
                                           description='UO simple model with parameter c, '
                                                       'attachment probability = p_attachment,'
                                                       'orientation variance = var_orientation, '
                                                       'orientation information = var_information')
        else:

            self.exp.load(self.name)
            self.n_replica = max(self.exp.get_index(self.name).get_level_values(id_exp_name))
            self.duration = max(self.exp.get_index(self.name).get_level_values(self.time_name))

    def run(self, para_value):

        self.para.change_parameter_values(para_value)
        self.name_column = str(self.para.get_parameter_tuple())

        self.res = []
        # a = self.para.var_orientation + self.para.p_attachment * self.para.c ** 2 * self.para.var_information
        # b = self.para.p_attachment*self.para.c*(2-self.para.c)
        print(self.name_column)

        for id_exp in range(1, self.n_replica + 1):
            self.id_exp = id_exp
            self.orientation = np.around(rd.uniform(-np.pi, np.pi), 3)
            # self.orientation = np.around(rd.normalvariate(0, np.sqrt(self.para.var_orientation)), 3)
            # self.orientation = 0
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
            theta_ant = rd.normalvariate(0, np.sqrt(self.para.var_information))

            dtheta = theta_ant - self.orientation
            self.orientation += self.para.c * dtheta

            self.last_attachment_time = self.t

        rho = rd.normalvariate(0, np.sqrt(self.para.var_orientation))
        self.orientation += rho

        self.res.append(np.around(self.orientation, 3))

    def write(self):
        self.exp.write(self.name)


class UOOutsideModel(BaseModels):
    def __init__(self, root, group, time0=-60, time1=200, n_replica=500, new=False):

        self.exp = ExperimentGroups(root, group)
        self.name = 'UOOutsideModel'
        parameter_names = ['c', 'var_orientation', 'var_information']

        BaseModels.__init__(self, parameter_names)

        self.time0 = time0
        self.time1 = time1
        self.n_replica = n_replica

        self.res = None

        self.init(new)

    def get_p_attachment_model(self):
        p_attachment = (1.05*np.log(self.t)-1.04)/10.
        return p_attachment

    def init(self, new):
        if new is True or not self.exp.is_name_existing(self.name):
            index = [(id_exp, t*100) for id_exp in range(1, self.n_replica+1) for t in range(self.time0, self.time1+1)]

            self.exp.add_new_empty_dataset(name=self.name, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO outside model',
                                           description='UO outside model with parameter c, '
                                                       'attachment probability = p_attachment,'
                                                       'orientation variance = var_orientation, '
                                                       'orientation information = var_information')
        else:

            self.exp.load(self.name)
            self.n_replica = max(self.exp.get_index(self.name).get_level_values(id_exp_name))
            self.time0 = min(self.exp.get_index(self.name).get_level_values(self.time_name))
            self.time1 = max(self.exp.get_index(self.name).get_level_values(self.time_name))

    def run(self, para_value):

        self.para.change_parameter_values(para_value)
        self.name_column = str(self.para.get_parameter_tuple())

        self.res = []
        print(self.name_column)

        for id_exp in range(1, self.n_replica + 1):
            self.id_exp = id_exp
            self.orientation = np.around(rd.uniform(-np.pi, np.pi), 3)
            self.res.append(self.orientation)
            self.t = self.time0
            self.last_attachment_time = 0

            for _ in range(self.time0+1, self.time1+1):
                self.step()

        self.exp.get_df(self.name)[self.name_column] = self.res

    def step(self):
        self.t += 1

        u = rd.random()
        p_attachment = self.get_p_attachment_model()

        if self.t >= 0:
            if u < p_attachment:
                theta_ant = rd.normalvariate(0, np.sqrt(self.para.var_information))

                dtheta = theta_ant - self.orientation
                self.orientation += self.para.c * dtheta

                self.last_attachment_time = self.t

        rho = rd.normalvariate(0, np.sqrt(self.para.var_orientation))
        self.orientation += rho

        self.res.append(np.around(self.orientation, 3))

    def write(self):
        self.exp.write(self.name)


class UORWModel(BaseModels):
    def __init__(self, root, group, duration=300, n_replica=500, new=False):

        self.exp = ExperimentGroups(root, group)
        self.name = 'UORWModel'
        parameter_names = ['c', 'p_attachment', 'd_orientation', 'd_information']

        BaseModels.__init__(self, parameter_names)

        self.duration = duration
        self.n_replica = n_replica

        self.res = None

        self.init(new)

    def init(self, new):
        if new is True or not self.exp.is_name_existing(self.name):
            index = [(id_exp, t*100) for id_exp in range(1, self.n_replica+1) for t in range(self.duration+1)]

            self.exp.add_new_empty_dataset(name=self.name, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO RW model', description='UO random walk model ')
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
            self.orientation = np.around(rd.uniform(-np.pi, np.pi), 3)
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

            if self.orientation < 0:
                self.orientation += self.para.c*self.para.d_information
            elif self.orientation > 0:
                self.orientation -= self.para.c*self.para.d_information

            self.last_attachment_time = self.t

        else:
            r = rd.random()
            if r < 0.5:
                self.orientation += self.para.d_orientation
            else:
                self.orientation -= self.para.d_orientation

        self.res.append(np.around(self.orientation, 3))

    def write(self):
        self.exp.write(self.name)


class UOSimpleExpModel(BaseModels):
    def __init__(self, root, group, duration=300, n_replica=500, new=False):

        self.exp = ExperimentGroups(root, group)
        self.name = 'UOSimpleExpModel'
        parameter_names = ['c', 'p_attachment', 'var_orientation', 'var_information']

        BaseModels.__init__(self, parameter_names)

        self.duration = duration
        self.n_replica = n_replica

        self.res = None

        self.init(new)

    def init(self, new):
        if new is True or not self.exp.is_name_existing(self.name):
            index = [(id_exp, t*100) for id_exp in range(1, self.n_replica+1) for t in range(self.duration+1)]

            self.exp.add_new_empty_dataset(name=self.name, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO simple exponential model',
                                           description='UO simple model with parameter c, '
                                                       'attachment probability = p_attachment,'
                                                       'orientation variance = var_orientation, '
                                                       'orientation information = var_information')
        else:

            self.exp.load(self.name)
            self.n_replica = max(self.exp.get_index(self.name).get_level_values(id_exp_name))
            self.duration = max(self.exp.get_index(self.name).get_level_values(self.time_name))

    def run(self, para_value):

        self.para.change_parameter_values(para_value)
        self.name_column = str(self.para.get_parameter_tuple())

        self.res = []
        # a = self.para.var_orientation + self.para.p_attachment * self.para.c ** 2 * self.para.var_information
        # b = self.para.p_attachment*self.para.c*(2-self.para.c)
        print(self.name_column)

        for id_exp in range(1, self.n_replica + 1):
            self.id_exp = id_exp
            self.orientation = np.around(rd.uniform(-np.pi, np.pi), 3)
            # self.orientation = np.around(rd.normalvariate(0, np.sqrt(self.para.var_orientation)), 3)
            # self.orientation = 0
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
            theta_ant = rd.normalvariate(0, np.sqrt(self.para.var_information))

            dtheta = theta_ant-self.orientation
            self.orientation += self.para.c*dtheta

            self.last_attachment_time = self.t

        rho = rd.normalvariate(0, np.sqrt(self.para.var_orientation))
        self.orientation += rho

        self.res.append(np.around(self.orientation, 3))

    def write(self):
        self.exp.write(self.name)


class UOConfidenceModel(BaseModels):
    def __init__(self, root, group, duration=300, n_replica=500, new=False):

        self.exp = ExperimentGroups(root, group)
        self.name = 'UOConfidenceModel'
        self.name_confidence = 'UOConfidenceModel_confidence'

        parameter_names = ['p_attachment', 'var_orientation', 'var_information']
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
            index = [(id_exp, t*100) for id_exp in range(1, self.n_replica+1) for t in range(self.duration+1)]

            self.exp.add_new_empty_dataset(name=self.name, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO confidence model',
                                           description='UO confidence model with parameter '
                                                       'attachment probability=p_attachment,'
                                                       'orientation variance = var_orientation, '
                                                       'orientation information = var_information')

            self.exp.add_new_empty_dataset(name=self.name_confidence, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='Confidence of the UO confidence model',
                                           description='Confidence of the UO confidence model with parameter '
                                                       'attachment probability=p_attachment,'
                                                       'orientation variance = var_orientation, '
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
            (self.para.p_attachment, self.para.var_orientation, self.para.var_information))

        self.res_orientation = []
        self.res_confidence = []
        print(self.name_column)

        for id_exp in range(1, self.n_replica + 1):
            self.id_exp = id_exp
            self.orientation = np.around(rd.uniform(-np.pi, np.pi), 3)
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
            theta_ant = rd.normalvariate(0, np.sqrt(self.para.var_information))

            dtheta = theta_ant-self.orientation

            self.orientation += self.c_ant*dtheta/(self.confidence+self.c_ant)
            self.confidence = 1/(self.para.var_orientation+1/(self.confidence+self.c_ant))
        else:
            self.confidence = 1/(self.para.var_orientation+1/self.confidence)

        rho = rd.normalvariate(0, np.sqrt(self.para.var_orientation))
        self.orientation += rho

        self.res_orientation.append(np.around(self.orientation, 3))
        self.res_confidence.append(np.around(self.confidence, 3))

    def write(self):
        self.exp.write([self.name, self.name_confidence])


class PersistenceModel(BaseModels):
    def __init__(self, root, group, duration=100, n_replica=500, new=False):

        self.exp = ExperimentGroups(root, group)
        self.name = 'PersistenceModel'
        parameter_names = ['var_orientation']

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
                                           label='Persistence model',
                                           description='Persistence model with parameter'
                                                       ' orientation variance = var_orientation')
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
            self.orientation = 0
            self.res.append(self.orientation)
            self.t = 0
            self.last_attachment_time = 0

            for t in range(1, self.duration + 1):
                self.step()

        self.exp.get_df(self.name)[self.name_column] = self.res

    def step(self):
        self.t += 1

        rho = rd.normalvariate(0, np.sqrt(self.para.var_orientation))
        self.orientation += rho

        self.res.append(np.around(self.orientation, 3))

    def write(self):
        self.exp.write(self.name)


class AnalyseUOModel(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'Models'

    def plot_simple_model_evol(self, suff=None, n=None, m=None):

        name = 'UOSimpleModel'
        self._plot_hist_evol(name, n, m, 'simple', suff)
        self._plot_var_evol(name, n, m, 'simple', suff)

    def plot_outside_model_evol(self, suff=None, n=None, m=None):

        name = 'UOOutsideModel'
        self._plot_hist_evol(name, n, m, 'outside', suff)
        self._plot_var_evol(name, n, m, 'outside', suff)

    def plot_rw_model_evol(self, suff=None, n=None, m=None):

        name = 'UORWModel'
        self._plot_hist_evol(name, n, m, 'simple', suff)
        self._plot_var_evol(name, n, m, 'simple', suff)

    def plot_confidence_model_evol(self, suff=None, n=None, m=None):

        name = 'UOConfidenceModel'
        self._plot_hist_evol(name, n, m, 'confidence', suff)
        self._plot_var_evol(name, n, m, 'confidence', suff)
        self._plot_confidence_evol(name+'_confidence', n, m, 'confidence', suff)

    def plot_persistence(self, suff=None):
        name = 'PersistenceModel'
        # self.__plot_cosinus_correlation_vs_length(name, suff)
        self.__plot_cosinus_correlation_vs_arclength(name, suff)

    def __plot_cosinus_correlation_vs_length(self, name, suff=None):

        self.exp.load(name)
        column_names = self.exp.get_data_object(name).get_column_names()

        n_replica = max(self.exp.get_df(name).index.get_level_values(id_exp_name))
        duration = max(self.exp.get_df(name).index.get_level_values('t'))

        speed = 4.66
        index_values = np.arange(duration + 1)/speed

        self.exp.add_new_empty_dataset('plot', index_names='lag', column_names=column_names,
                                       index_values=index_values,
                                       fill_value=0, category=self.category, replace=True)

        for column_name in column_names:
            df = pd.DataFrame(self.exp.get_df(name)[column_name])

            for id_exp in range(1, max(df.index.get_level_values(id_exp_name))+1):
                df2 = df.loc[id_exp, :]

                orientations = np.array(df2)
                res = np.zeros(len(orientations))
                weight = np.zeros(len(orientations))

                for i in range(1, len(orientations)):
                    res[:-i] += np.cos(orientations[i] - orientations[i:]).ravel()
                    weight[:-i] += 1.

                res /= weight

                self.exp.get_df('plot')[column_name] += res

        self.exp.get_data_object('plot').df /= float(n_replica)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('plot'))
        fig, ax = plotter.plot(xlabel='Distance along trajectory (cm)', ylabel='Cosine correlation', marker=None)
        ax.axhline(0, ls='--', c='grey')
        ax.grid()

        if suff is None:
            plotter.save(fig, name=name+'_length')
        else:
            plotter.save(fig, name=name+'_length_'+suff)

    def __plot_cosinus_correlation_vs_arclength(self, name, suff=None):

        self.exp.load(name)
        column_names = self.exp.get_data_object(name).get_column_names()

        radius = 0.85

        dtheta = 1.
        index_values = np.arange(0, 500, dtheta)

        self.exp.add_new_empty_dataset('plot', index_names='lag', column_names=column_names,
                                       index_values=index_values,
                                       fill_value=0, category=self.category, replace=True)

        for column_name in column_names:
            df = pd.DataFrame(self.exp.get_df(name)[column_name])
            norm = np.zeros(len(index_values))

            for id_exp in range(1, max(df.index.get_level_values(id_exp_name))+1):
                df2 = df.loc[id_exp, :]
                orientations = np.array(df2).ravel()
                d_orientations = angle_distance(orientations[1:], orientations[:-1])
                arclength = np.cumsum(np.abs(d_orientations)) * radius

                orientations2 = np.zeros(len(index_values))
                idx = 0
                for i, arc in enumerate(arclength):
                    idx = get_index_interval_containing(arc, index_values)
                    orientations2[idx:] = orientations[i]

                orientations2 = orientations2[:idx+1]

                corr = np.zeros(len(orientations2))
                weight = np.zeros(len(orientations2))

                for i in range(1, len(orientations2)):
                    corr[:-i] += np.cos(orientations2[i] - orientations2[i:]).ravel()
                    weight[:-i] += 1.

                corr2 = np.zeros(len(index_values))
                corr2[:len(corr)] = corr / weight

                norm[:len(orientations2)] += 1
                self.exp.get_df('plot')[column_name] += corr2

            self.exp.get_df('plot')[column_name] /= norm

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('plot'))
        fig, ax = plotter.plot(xlabel='Arclength along trajectory (cm)', ylabel='Cosine correlation', marker=None)
        ax.axhline(0, ls='--', c='grey')
        ax.set_xlim((0, 12.5))
        ax.set_ylim((-0.1, 1.1))
        ax.grid()

        if suff is None:
            plotter.save(fig, name=name+'_arclength')
        else:
            plotter.save(fig, name=name+'_arclength_'+suff)

    def _plot_confidence_evol(self, name, n=None, m=None, model=None, suff=None):

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
            plotter.plot(xlabel='orientation', ylabel='PDF', preplot=(fig, ax0),
                         title=column_name, label='confidence mean', marker=None)

            p_attachment = float(column_name.split(',')[0][1:])
            var_orientation = float(column_name.split(',')[1])
            var_info = float(column_name.split(',')[-1][:-1])

            c_ant = 1 / var_info
            b = var_orientation * c_ant
            c_obj = (-b + np.sqrt(b*(b + 4))) / (2*var_orientation)

            ax0.axhline(c_ant, ls='--', label='c_ant', c='g')
            ax0.axhline(c_obj, ls='--', label='c_obj', c='b')

            ax0.axhline(c_obj*p_attachment, ls='--', label='c_obj*p_att', c='y')
            ax0.set_ylim((0, max([np.max(df2/float(len(list_id_exp))), c_ant])+1.))

            ax0.legend()

        if model == 'simple':
            fig.suptitle(r"Simple model, parameters $(c, p_{att}, \sigma_{orient}, \sigma_{info})$",
                         fontsize=15)
        elif model == 'confidence':
            fig.suptitle(r"Confidence model, parameters $(p_{att}, \sigma_{orient}, \sigma_{info})$",
                         fontsize=15)

        if suff is None:
            plotter.save(fig, name=name)
        else:
            plotter.save(fig, name=name+'_'+suff)

    def _plot_hist_evol(self, name, n=None, m=None, model=None, suff=None):

        experimental_name = 'food_direction_error_hist_evol'
        experimental_name_attach = 'food_direction_error_hist_evol_around_first_attachment'
        self.exp.load([name, experimental_name, experimental_name_attach])

        self.exp.load(name)
        self.exp.get_data_object(name).df = np.abs(self.exp.get_df(name))

        time_name = 't'
        column_names = self.exp.get_data_object(name).get_column_names()

        dx = 0.25
        start_time_intervals = np.arange(0, 4., dx)*60*100
        end_time_intervals = start_time_intervals + dx*60*100*2

        dtheta = np.pi / 25.
        bins = np.arange(0, np.pi, dtheta)

        if n is None:
            n = int(np.floor(np.sqrt(len(column_names))))
            m = int(np.ceil(len(column_names) / n))

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(name))
        fig, ax = plotter.create_plot(figsize=(4 * m, 4 * n), nrows=n, ncols=m, top=0.85, bottom=0.1, left=0.05)

        plotter_exp = Plotter(root=self.exp.root, obj=self.exp.get_data_object(experimental_name))
        fig2, ax2 = plotter_exp.create_plot(figsize=(4 * m, 4 * n), nrows=n, ncols=m, top=0.85, bottom=0.1, left=0.05)

        plotter_exp2 = Plotter(root=self.exp.root, obj=self.exp.get_data_object(experimental_name_attach))
        fig3, ax3 = plotter.create_plot(figsize=(4 * m, 4 * n), nrows=n, ncols=m, top=0.85, bottom=0.1, left=0.05)

        for k, column_name in enumerate(column_names):

            hist_name = self.exp.hist1d_evolution(name_to_hist=name, start_index_intervals=start_time_intervals,
                                                  end_index_intervals=end_time_intervals, bins=bins,
                                                  index_name=time_name, column_to_hist=column_name, replace=True)
            # self.exp.get_df(hist_name).index = self.exp.get_index(hist_name)**2

            i = int(np.floor(k / m))
            j = k % m
            if isinstance(ax, np.ndarray):
                if len(ax.shape) == 1:
                    ax0 = ax[k]
                    ax20 = ax2[k]
                    ax30 = ax3[k]
                else:
                    ax0 = ax[i, j]
                    ax20 = ax2[i, j]
                    ax30 = ax3[i, j]
            else:
                ax0 = ax
                ax20 = ax2
                ax30 = ax3

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
            plotter.plot(xlabel='orientation', ylabel='PDF',
                         normed=True, preplot=(fig, ax0), title=column_name)

            plotter_exp.plot(xlabel='orientation', ylabel='PDF', marker='',
                             normed=True, preplot=(fig2, ax20), title=column_name)

            plotter_exp2.plot(xlabel='orientation', ylabel='PDF', marker='',
                              normed=True, preplot=(fig3, ax30), title=column_name)

            if model == 'simple':

                c = float(column_name.split(',')[0][1:])
                p_attach = float(column_name.split(',')[1])
                var_orientation = float(column_name.split(',')[2])
                var_info = float(column_name.split(',')[3][:-1])

                b = var_orientation+p_attach*c**2*var_info
                a = 1-p_attach*c*(2-c)
                r = b/(1-a)

                x = self.exp.get_df(hist_name).index

                ax0.plot(x, 2*scs.norm.pdf(x, scale=np.sqrt(r)))
                ax20.plot(x, 2*scs.norm.pdf(x, scale=np.sqrt(r)))
                ax30.plot(x, 2*scs.norm.pdf(x, scale=np.sqrt(r)))

        if model == 'simple':
            fig.suptitle(r"$(c, p_{att}, \sigma_{orient}, \sigma_{info})$ = ",
                         fontsize=15)
            fig2.suptitle(r"$(c, p_{att}, \sigma_{orient}, \sigma_{info})$ = ",
                          fontsize=15)
            fig3.suptitle(r"$(c, p_{att}, \sigma_{orient}, \sigma_{info})$ = ",
                          fontsize=15)
        if model == 'outside':
            fig.suptitle(r"$(c, \sigma_{orient}, \sigma_{info})$ = ",
                         fontsize=15)
        elif model == 'confidence':
            fig.suptitle(r"Confidence model, parameters $(p_{att}, \sigma_{orient}, \sigma_{info})$",
                         fontsize=15)

        if suff is None:
            fig_name = name + '_hist'
        else:
            fig_name = name + '_hist_' + suff
        plotter.save(fig, name=fig_name)
        plotter.save(fig2, name=fig_name+'_experiment')
        plotter.save(fig3, name=fig_name+'_experiment_around_first_attachment')

        self.exp.remove_object(name)

    def plot_indiv_hist_evol(self, name, column_num, model='simple', suff=None):

        experimental_name = 'food_direction_error_hist_evol'
        experimental_name_attach = 'food_direction_error_hist_evol_around_first_attachment'
        self.exp.load([name, experimental_name, experimental_name_attach])

        self.exp.load(name)
        self.exp.get_data_object(name).df = np.abs(self.exp.get_df(name))

        time_name = 't'
        column_name = self.exp.get_columns(name)[column_num]

        dx = 0.25
        start_time_intervals = np.arange(0, 4., dx)*60*100
        end_time_intervals = start_time_intervals + dx*60*100*2

        dtheta = np.pi / 25.
        bins = np.arange(0, np.pi, dtheta)

        n = m = 4

        plotter_to_save = Plotter(root=self.exp.root, obj=self.exp.get_data_object(name))
        fig, ax = plotter_to_save.create_plot(figsize=(4 * m, 4 * n), nrows=n, ncols=m, top=0.9, bottom=0.05, left=0.05)

        hist_name = self.exp.hist1d_evolution(name_to_hist=name, start_index_intervals=start_time_intervals,
                                              end_index_intervals=end_time_intervals, bins=bins,
                                              index_name=time_name, column_to_hist=column_name, replace=True)
        column_names = self.exp.get_columns(hist_name)

        for k, column_name2 in enumerate(column_names):

            i = int(np.floor(k / m))
            j = k % m
            ax0 = ax[i, j]

            self.exp.add_new_dataset_from_df(self.exp.get_df(hist_name)[column_name2], 'temp', replace=True)
            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
            plotter.plot(xlabel='orientation', ylabel='PDF', c='k', marker='', ls='--',
                         normed=True, preplot=(fig, ax0), label='Model', title=r't$\in$' + column_name2)

            if column_name2 in self.exp.get_columns(experimental_name):
                self.exp.add_new_dataset_from_df(self.exp.get_df(experimental_name)[column_name2], 'temp', replace=True)
                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
                plotter.plot(xlabel='orientation', ylabel='PDF', marker='', c='red',
                             normed=True, preplot=(fig, ax0), label='Exp', title=r't$\in$' + column_name2)

            if column_name2 in self.exp.get_columns(experimental_name_attach):
                self.exp.add_new_dataset_from_df(
                    self.exp.get_df(experimental_name_attach)[column_name2], 'temp', replace=True)
                plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object('temp'))
                plotter.plot(xlabel='orientation', ylabel='PDF', c='orange', marker='',
                             normed=True, preplot=(fig, ax0), label='Aligned Exp', title=r't$\in$' + column_name2)

            if model == 'simple':

                c = float(column_name.split(',')[0][1:])
                p_attach = float(column_name.split(',')[1])
                var_orientation = float(column_name.split(',')[2])
                var_info = float(column_name.split(',')[3][:-1])
                var0 = np.pi**2/3.

                b = var_orientation+p_attach*c**2*var_info
                a = 1-p_attach*c*(2-c)
                r = b/(1-a)

                t0 = float(column_name2.split(',')[0][1:])
                t1 = float(column_name2.split(',')[1][:-1])
                t = (t0+t1)/2.
                var = a**t*(var0-r)+r

                x = self.exp.get_df(hist_name).index

                ax0.plot(x, 2*scs.norm.pdf(x, scale=np.sqrt(var)), label='Theory', c='w', ls='--')
            ax0.legend()

        if suff is None:
            fig_name = name + 'indiv_hist'
        else:
            fig_name = name + 'indiv_hist_' + suff
        plotter_to_save.save(fig, name=fig_name)
        self.exp.remove_object(name)

    def _plot_var_evol(self, name, n=None, m=None, model=None, suff=None):

        experimental_name = 'food_direction_error_var_evol'
        experimental_name_attach = 'food_direction_error_var_evol_around_first_attachment'
        self.exp.load([name, experimental_name, experimental_name_attach])

        time_name = 't'
        column_names = self.exp.get_data_object(name).get_column_names()

        dx = 0.1
        dx2 = 0.01
        start_time_intervals = np.arange(0, 3., dx2)*60*100
        end_time_intervals = start_time_intervals + dx*60*100*2

        if n is None:
            n = int(np.floor(np.sqrt(len(column_names))))
            m = int(np.ceil(len(column_names) / n))

        plotter_experiment = Plotter(
            root=self.exp.root, obj=self.exp.get_data_object(experimental_name), category=self.category)
        plotter_experiment_attach = Plotter(
            root=self.exp.root, obj=self.exp.get_data_object(experimental_name_attach), category=self.category)
        fig, ax = plotter_experiment.create_plot(figsize=(4 * m, 4 * n), nrows=n, ncols=m, top=0.85, bottom=0.01)

        for k, column_name in enumerate(column_names):

            var_name = self.exp.variance_evolution(name_to_var=name,  start_index_intervals=start_time_intervals,
                                                   end_index_intervals=end_time_intervals, index_name=time_name,
                                                   column_to_var=column_name, replace=True)
            self.exp.get_df(var_name).index = self.exp.get_index(var_name)/100.

            i = int(np.floor(k / m))
            j = k % m
            if isinstance(ax, np.ndarray):
                if len(ax.shape) == 1:
                    ax0 = ax[k]
                else:
                    ax0 = ax[i, j]
            else:
                ax0 = ax

            if model == 'simple':
                c = float(column_name.split(',')[0][1:])
                p_attach = float(column_name.split(',')[1])
                var_orientation = float(column_name.split(',')[2])
                var_info = float(column_name.split(',')[3][:-1])

                def variance(t):
                    b = var_orientation+p_attach*c**2*var_info
                    a = 1-p_attach*c*(2-c)
                    r = b/(1-a)
                    s = np.pi**2/3.
                    return a**t*(s-r)+r

                t_tab = np.array(self.exp.get_df(var_name).index)
                ax0.plot(t_tab, variance(t_tab), label='Theory')

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(var_name))
            plotter.plot(xlabel='Time', ylabel='Variance', preplot=(fig, ax0), title=column_name, label='Model')
            plotter_experiment.plot(
                xlabel='Time', ylabel='Variance', preplot=(fig, ax0), c='grey', title=column_name, label='exp')
            plotter_experiment_attach.plot(
                xlabel='Time', ylabel='Variance', preplot=(fig, ax0), c='w', title=column_name, label='exp 2')

            ax0.legend()
            plotter.draw_vertical_line(ax0)

        if model == 'simple':
            fig.suptitle(r"$(c, p_{att}, \sigma_{orient}, \sigma_{info})$ = ", fontsize=15)
        elif model == 'outside':
            fig.suptitle(r"$(c, \sigma_{orient}, \sigma_{info})$ = ", fontsize=15)
        elif model == 'confidence':
            fig.suptitle(r"Confidence model, parameters $(p_{att}, \sigma_{orient}, \sigma_{info})$", fontsize=15)

        if suff is None:
            plotter_experiment.save(fig, name=name+'_var')
        else:
            plotter_experiment.save(fig, name=name+'_var_'+suff)

    def plot_hist_model_pretty(self, name, n=None, m=None, display_title=True):

        experimental_name = 'food_direction_error_hist_evol'
        experimental_name_attach = 'food_direction_error_hist_evol_around_first_attachment'
        self.exp.load([name, experimental_name, experimental_name_attach])

        self.exp.load(name)
        self.exp.get_data_object(name).df = np.abs(self.exp.get_df(name))

        time_name = 't'
        column_names = self.exp.get_data_object(name).get_column_names()

        dx = 0.25
        start_time_intervals = np.arange(0, 3., dx)*60*100
        end_time_intervals = start_time_intervals + dx*60*100*2

        dtheta = np.pi / 25.
        bins = np.arange(0, np.pi, dtheta)

        if n is None:
            n = int(np.floor(np.sqrt(len(column_names))))
            m = int(np.ceil(len(column_names) / n))

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(name))
        if name == 'UOSimpleModel':
            fig, ax = plotter.create_plot(figsize=(4 * m, 4 * n), nrows=n, ncols=m, top=0.98, bottom=0.12, left=0.1)
        else:
            fig, ax = plotter.create_plot(figsize=(4*m, 4*n+1), nrows=n, ncols=m, top=0.95, bottom=0.08, left=0.06)

        for k, column_name in enumerate(column_names):

            hist_name = self.exp.hist1d_evolution(name_to_hist=name, start_index_intervals=start_time_intervals,
                                                  end_index_intervals=end_time_intervals, bins=bins,
                                                  index_name=time_name, column_to_hist=column_name, replace=True)

            i = int(np.floor(k / m))
            j = k % m
            if isinstance(ax, np.ndarray):
                if len(ax.shape) == 1:
                    ax0 = ax[k]
                else:
                    ax0 = ax[i, j]
            else:
                ax0 = ax

            if display_title:
                c = float(column_name.split(',')[0][1:])
                title = 'c = ' + str(c)
            else:
                title = ''
            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(hist_name))
            plotter.plot(xlabel=r'$\theta$', ylabel='PDF', title=title, normed=True, preplot=(fig, ax0),
                         display_legend=False)
            if k == 0:
                plotter.draw_legend(ax0, ncol=2)

        fig_name = name + '_hist_pretty'
        plotter.save(fig, name=fig_name)

        self.exp.remove_object(name)

    def plot_var_model_pretty(self, name, n=None, m=None, display_title=True):

        name_exp_variance = 'food_direction_error_var_evol_around_first_attachment'
        name_exp_fisher_info = 'fisher_info_evol_around_first_attachment'
        self.exp.load([name, name_exp_variance, name_exp_fisher_info])

        time_name = 't'
        column_names = self.exp.get_data_object(name).get_column_names()

        dx = 0.1
        dx2 = 0.01
        start_time_intervals = np.arange(0, 3., dx2)*60*100
        end_time_intervals = start_time_intervals + dx*60*100*2

        if n is None:
            n = int(np.floor(np.sqrt(len(column_names))))
            m = int(np.ceil(len(column_names) / n))

        plotter_exp_variance = Plotter(
            root=self.exp.root, obj=self.exp.get_data_object(name_exp_variance), category=self.category)
        plotter_exp_fisher_info = Plotter(
            root=self.exp.root, obj=self.exp.get_data_object(name_exp_fisher_info), category=self.category)
        if name == 'UOSimpleModel':
            fig, ax = plotter_exp_variance.create_plot(figsize=(4 * m, 4 * n), nrows=n, ncols=m,
                                                       top=0.98, bottom=0.12, left=0.1)
            fig2, ax2 = plotter_exp_fisher_info.create_plot(figsize=(4 * m, 4 * n), nrows=n, ncols=m,
                                                            top=0.98, bottom=0.12, left=0.1)
        else:
            fig, ax = plotter_exp_variance.create_plot(figsize=(4*m, 4*n+1), nrows=n, ncols=m,
                                                       top=0.95, bottom=0.08, left=0.06)
            fig2, ax2 = plotter_exp_fisher_info.create_plot(figsize=(4 * m, 4 * n+1), nrows=n, ncols=m,
                                                            top=0.97, bottom=0.1, left=0.05)

        for k, column_name in enumerate(column_names):

            c = float(column_name.split(',')[0][1:])
            var_name = self.exp.variance_evolution(name_to_var=name,  start_index_intervals=start_time_intervals,
                                                   end_index_intervals=end_time_intervals, index_name=time_name,
                                                   column_to_var=column_name, replace=True)
            self.exp.get_df(var_name).index = self.exp.get_index(var_name)/100.

            i = int(np.floor(k / m))
            j = k % m
            if isinstance(ax, np.ndarray):
                if len(ax.shape) == 1:
                    ax0 = ax[k]
                    ax02 = ax2[k]
                else:
                    ax0 = ax[i, j]
                    ax02 = ax2[i, j]
            else:
                ax0 = ax
                ax02 = ax2

            plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(var_name))
            if display_title:
                title = 'c = ' + str(c)
            else:
                title = ''

            plotter.plot(
                xlabel='Time (s)', ylabel='Variance', preplot=(fig, ax0), label='Model', marker='', title=title)
            plotter_exp_variance.plot(
                xlabel='Time (s)', ylabel='Variance', preplot=(fig, ax0),
                c='w', label='Experiment', marker='', title=title)

            plotter.plot(xlabel='Time (s)', ylabel='Fisher information', fct=lambda a: 1/a,
                         preplot=(fig2, ax02), label='Model', marker='', title=title)
            plotter_exp_variance.plot(
                xlabel='Time (s)', ylabel='Fisher information', fct=lambda a: 1/a, preplot=(fig2, ax02),
                c='w', label='Experiment', marker='', title=title)

            ax0.legend()
            plotter.draw_vertical_line(ax0)
            ax02.legend()
            plotter.draw_vertical_line(ax02)

        plotter_exp_variance.save(fig, name=name+'_var_pretty')
        plotter_exp_variance.save(fig2, name=name+'_fisher_info_pretty')
