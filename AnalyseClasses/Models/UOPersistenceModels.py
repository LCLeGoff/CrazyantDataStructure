import random as rd

import numpy as np

from AnalyseClasses.Models.UOModels import BaseModels
from DataStructure.VariableNames import id_exp_name
from ExperimentGroups import ExperimentGroups
from Tools.MiscellaneousTools import Geometry as Geo


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


class WrappedPersistenceModel(BaseModels):
    def __init__(self, root, group, duration=100, n_replica=500, new=False):

        self.exp = ExperimentGroups(root, group)
        self.name = 'WrappedPersistenceModel'
        parameter_names = ['kappa_orientation']

        BaseModels.__init__(self, parameter_names)

        self.duration = duration
        self.n_replica = n_replica

        self.res = None
        self.rd = None

        self.init(new)

    def init(self, new):
        if new is True or not self.exp.is_name_existing(self.name):
            index = [(id_exp, t) for id_exp in range(1, self.n_replica+1) for t in range(self.duration+1)]

            self.exp.add_new_empty_dataset(name=self.name, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='Wrapped Persistence model',
                                           description='Wrapped Persistence model with parameter'
                                                       ' orientation variance = 1/kappa_orientation')
        else:

            self.exp.load(self.name)
            self.n_replica = max(self.exp.get_index(self.name).get_level_values(id_exp_name))
            self.duration = max(self.exp.get_index(self.name).get_level_values(self.time_name))

    def run(self, para_value):

        self.para.change_parameter_values(para_value)
        self.name_column = str(self.para.get_parameter_tuple())

        self.res = []
        self.rd = np.random.vonmises(mu=0, kappa=self.para.kappa_orientation, size=(self.duration+1)*self.n_replica)
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

        idx = (self.id_exp-1)*(self.duration+1) + self.t
        rho = self.rd[idx]
        self.orientation = Geo.angle_sum(self.orientation, rho)

        self.res.append(np.around(self.orientation, 3))

    def write(self):
        self.exp.write(self.name)


class UniformPersistenceModel(BaseModels):
    def __init__(self, root, group, duration=100, n_replica=500, new=False):

        self.exp = ExperimentGroups(root, group)
        self.name = 'UniformPersistenceModel'
        parameter_names = ['c', 'var0']

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
                                           label='Uniform persistence model',
                                           description='Uniform persistence model with parameter'
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
            self.orientation = rd.normalvariate(0, np.sqrt(self.para.var0))
            self.res.append(self.orientation)
            self.t = 0
            self.last_attachment_time = 0

            for t in range(1, self.duration + 1):
                self.step()

        self.exp.get_df(self.name)[self.name_column] = self.res

    def step(self):
        self.t += 1

        rho = rd.uniform(-np.pi, np.pi)
        # self.orientation = Geo.angle_sum(self.orientation, rho)
        self.orientation += rho*self.para.c
        # self.orientation += self.para.c*(rho-self.orientation)

        self.res.append(np.around(self.orientation, 3))

    def write(self):
        self.exp.write(self.name)
