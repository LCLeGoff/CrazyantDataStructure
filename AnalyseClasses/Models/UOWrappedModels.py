import random as rd

import numpy as np

from AnalyseClasses.Models.UOModels import BaseModels
from DataStructure.VariableNames import id_exp_name
from ExperimentGroups import ExperimentGroups
from Tools.MiscellaneousTools import Geometry as Geo, Fits


class UOWrappedSimpleModel(BaseModels):
    def __init__(self, root, group, duration=300, n_replica=500, new=False):

        self.exp = ExperimentGroups(root, group)
        self.name_orient = 'UOWrappedSimpleModel'
        self.name_attachments = 'UOWrappedSimpleModelAttachments'
        parameter_names = ['c', 'p_attachment', 'kappa_orientation', 'kappa_information']

        BaseModels.__init__(self, parameter_names)

        self.duration = duration
        self.n_replica = n_replica

        self.res_orient = None
        self.res_attachments = None
        self.rd_orientation = None
        self.rd_info = None

        self.init(new)

    def init(self, new):
        if new is True or not self.exp.is_name_existing(self.name_orient):
            index = [(id_exp, t*100) for id_exp in range(1, self.n_replica+1) for t in range(self.duration+1)]

            self.exp.add_new_empty_dataset(name=self.name_orient, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO simple model',
                                           description='UO simple model with parameter c, '
                                                       'attachment probability = p_attachment,'
                                                       'orientation variance = 1/kappa_orientation, '
                                                       'orientation information = 1/kappa_information')

            self.exp.add_new_empty_dataset(name=self.name_attachments, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO simple model',
                                           description='Attachment instants of UO simple model with parameter c, '
                                                       'attachment probability = p_attachment,'
                                                       'orientation variance = 1/kappa_orientation, '
                                                       'orientation information = 1/kappa_information')
        else:

            self.exp.load(self.name_orient)
            self.exp.load(self.name_attachments)
            self.n_replica = max(self.exp.get_index(self.name_orient).get_level_values(id_exp_name))
            self.duration = max(self.exp.get_index(self.name_orient).get_level_values(self.time_name))

    def run(self, para_value):

        self.para.change_parameter_values(para_value)
        self.name_column = str(self.para.get_parameter_tuple())

        self.rd_orientation = np.random.vonmises(
            mu=0, kappa=self.para.kappa_orientation, size=(self.duration+1)*self.n_replica)
        self.rd_info = np.random.vonmises(
            mu=0, kappa=self.para.kappa_information, size=(self.duration+1)*self.n_replica)

        self.res_orient = []
        self.res_attachments = []
        print(self.name_column)

        for id_exp in range(1, self.n_replica + 1):
            self.id_exp = id_exp
            self.orientation = np.around(rd.uniform(-np.pi, np.pi), 3)
            self.res_orient.append(self.orientation)
            self.res_attachments.append(0)
            self.t = 0
            self.last_attachment_time = 0

            for t in range(1, self.duration + 1):
                self.step()

        self.exp.get_df(self.name_orient)[self.name_column] = self.res_orient
        self.exp.get_df(self.name_attachments)[self.name_column] = self.res_attachments

    def step(self):
        self.t += 1

        u = rd.random()
        p_attachment = self.get_p_attachment()

        idx = (self.id_exp-1)*(self.duration+1) + self.t
        if u < p_attachment:
            theta_ant = self.rd_info[idx]
            dtheta = Geo.angle_distance(self.para.c * theta_ant, self.para.c * self.orientation)
            self.orientation = Geo.angle_sum(self.orientation, dtheta)

            self.last_attachment_time = self.t
            self.res_attachments.append(1)
        else:
            self.res_attachments.append(0)

            rho = self.rd_orientation[idx]
            self.orientation = Geo.angle_sum(self.orientation, rho)

        self.res_orient.append(np.around(self.orientation, 3))

    def write(self):
        self.exp.write(self.name_orient)
        self.exp.write(self.name_attachments)


class UOWrappedOutInModel(BaseModels):
    def __init__(self, root, group, duration=300, n_replica=500, new=False, suff=None):

        self.exp = ExperimentGroups(root, group)
        self.name_orient = 'UOWrappedOutInModel'
        if suff is not None:
            self.name_orient += '_'+suff
        parameter_names = ['c_outside', 'c_inside', 'p_outside', 'p_inside', 'kappa_orientation', 'kappa_information']

        BaseModels.__init__(self, parameter_names)

        self.duration = duration
        self.n_replica = n_replica

        self.res_orient = None
        self.rd_orientation = None
        self.rd_info = None
        self.rd_attachment = None
        self.rd_inside = None

        self.init(new)

    def init(self, new):
        if new is True or not self.exp.is_name_existing(self.name_orient):
            index = [(id_exp, t*100) for id_exp in range(1, self.n_replica+1) for t in range(self.duration+1)]

            self.exp.add_new_empty_dataset(name=self.name_orient, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO simple model',
                                           description='UO simple model with parameter c_inside, c_outside, '
                                                       'outside attachment probability = p_outside,'
                                                       'inside attachment probability = p_inside,'
                                                       'orientation variance = 1/kappa_orientation, '
                                                       'orientation information = 1/kappa_information')
        else:

            self.exp.load(self.name_orient)
            self.n_replica = max(self.exp.get_index(self.name_orient).get_level_values(id_exp_name))
            self.duration = max(self.exp.get_index(self.name_orient).get_level_values(self.time_name))

    def run(self, para_value):

        self.para.change_parameter_values(para_value)
        self.name_column = str(self.para.get_parameter_tuple())

        self.rd_orientation = np.random.vonmises(
            mu=0, kappa=self.para.kappa_orientation, size=(self.duration+1)*self.n_replica)
        self.rd_info = np.random.vonmises(
            mu=0, kappa=self.para.kappa_information, size=(self.duration+1)*self.n_replica)
        self.rd_inside = np.random.uniform(low=-np.pi, high=np.pi, size=(self.duration+1)*self.n_replica)
        self.rd_attachment = np.random.uniform(size=(self.duration+1)*self.n_replica)

        self.res_orient = []
        print(self.name_column)

        for id_exp in range(1, self.n_replica + 1):
            self.id_exp = id_exp
            self.orientation = np.around(rd.uniform(-np.pi, np.pi), 3)
            self.res_orient.append(self.orientation)
            self.t = 0
            self.last_attachment_time = 0

            for t in range(1, self.duration + 1):
                self.step()

        self.exp.get_df(self.name_orient)[self.name_column] = self.res_orient

    def step(self):
        self.t += 1
        idx = (self.id_exp-1)*(self.duration+1) + self.t

        u = self.rd_attachment[idx]
        p_outside = self.para.p_outside
        p_inside = self.para.p_inside

        if u < p_outside:
            theta_ant = self.rd_info[idx]
            dtheta = Geo.angle_distance(self.para.c_outside * theta_ant, self.para.c_outside * self.orientation)
            self.orientation = Geo.angle_sum(self.orientation, dtheta)

            self.last_attachment_time = self.t
        elif p_outside < u < p_outside+p_inside:
            theta_ant = self.rd_inside[idx]
            dtheta = Geo.angle_distance(self.para.c_inside * theta_ant, self.para.c_inside * self.orientation)
            self.orientation = Geo.angle_sum(self.orientation, dtheta)

            self.last_attachment_time = self.t
        else:
            rho = self.rd_orientation[idx]
            self.orientation = Geo.angle_sum(self.orientation, rho)

        self.res_orient.append(np.around(self.orientation, 3))

    def write(self):
        self.exp.write(self.name_orient)


class UOWrappedOutInModel2(BaseModels):
    def __init__(self, root, group, duration=300, n_replica=500, new=False, suff=None):

        self.exp = ExperimentGroups(root, group)
        self.name_orient = 'UOWrappedOutInModel2'
        if suff is not None:
            self.name_orient += '_'+suff
        parameter_names = ['c_outside', 'c_inside', 'kappa_orientation', 'kappa_information']

        BaseModels.__init__(self, parameter_names)

        self.duration = duration
        self.n_replica = n_replica

        self.res_orient = None
        self.rd_orientation = None
        self.rd_info = None
        self.rd_attachment = None
        self.rd_inside = None

        self.init(new)

        x = np.arange(self.n_replica + 1)
        self.q2 = Fits.exp_fct(x, -0.062, 3.1, 0.64)

        self.p_outside = 0.057*(x*0+1)
        self.p_inside = self.p_outside*self.q2

    def get_p_inside(self):
        return self.p_inside[self.t]

    def get_p_outside(self):
        return self.p_outside[self.t]

    # @staticmethod
    # def get_c_outside():
    #     return max(0, 1-np.abs(np.random.laplace(scale=0.1)))

    def init(self, new):
        if new is True or not self.exp.is_name_existing(self.name_orient):
            index = [(id_exp, t*100) for id_exp in range(1, self.n_replica+1) for t in range(self.duration+1)]

            self.exp.add_new_empty_dataset(name=self.name_orient, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO simple model',
                                           description='UO simple model with parameter c_inside, c_outside, '
                                                       'outside attachment probability = p_outside,'
                                                       'inside attachment probability = p_inside,'
                                                       'orientation variance = 1/kappa_orientation, '
                                                       'orientation information = 1/kappa_information')
        else:

            self.exp.load(self.name_orient)
            self.n_replica = max(self.exp.get_index(self.name_orient).get_level_values(id_exp_name))
            self.duration = max(self.exp.get_index(self.name_orient).get_level_values(self.time_name))

    def run(self, para_value):

        self.para.change_parameter_values(para_value)
        self.name_column = str(self.para.get_parameter_tuple())

        if self.para.kappa_orientation == np.inf:
            self.rd_orientation = np.zeros((self.duration+1)*self.n_replica)
        else:
            self.rd_orientation = np.random.vonmises(
                mu=0, kappa=self.para.kappa_orientation, size=(self.duration+1)*self.n_replica)
        self.rd_info = np.random.vonmises(
            mu=0, kappa=self.para.kappa_information, size=(self.duration+1)*self.n_replica)
        self.rd_inside = np.random.uniform(low=-np.pi, high=np.pi, size=(self.duration+1)*self.n_replica)
        self.rd_attachment = np.random.uniform(size=(self.duration+1)*self.n_replica)

        self.res_orient = []
        print(self.name_column)

        for id_exp in range(1, self.n_replica + 1):
            self.id_exp = id_exp
            self.orientation = np.around(rd.uniform(-np.pi, np.pi), 3)
            self.res_orient.append(self.orientation)
            self.t = 0
            self.last_attachment_time = 0

            for t in range(1, self.duration + 1):
                self.step()

        self.exp.get_df(self.name_orient)[self.name_column] = self.res_orient

    def step(self):
        self.t += 1
        idx = (self.id_exp-1)*(self.duration+1) + self.t

        u = self.rd_attachment[idx]
        p_outside = self.get_p_outside()
        p_inside = self.get_p_inside()

        # c = self.get_c_outside()

        if u < p_outside:
            theta_ant = self.rd_info[idx]
            dtheta = Geo.angle_distance(self.para.c_outside*theta_ant, self.para.c_outside*self.orientation)
            self.orientation = Geo.angle_sum(self.orientation, dtheta)

            self.last_attachment_time = self.t
        elif p_outside < u < p_outside+p_inside:
            theta_ant = self.rd_inside[idx]
            dtheta = Geo.angle_distance(self.para.c_inside*theta_ant, self.para.c_inside*self.orientation)
            self.orientation = Geo.angle_sum(self.orientation, dtheta)

            self.last_attachment_time = self.t
        else:
            rho = self.rd_orientation[idx]
            self.orientation = Geo.angle_sum(self.orientation, rho)

        self.res_orient.append(np.around(self.orientation, 3))

    def write(self):
        self.exp.write(self.name_orient)


class UOWrappedOutInModel3(BaseModels):
    def __init__(self, root, group, duration=300, n_replica=500, new=False, suff=None, fps=1):

        self.exp = ExperimentGroups(root, group)
        self.name_orient = 'UOWrappedOutInModel3'
        if suff is not None:
            self.name_orient += '_'+suff
        parameter_names = ['c_outside', 'c_inside', 'kappa_orientation', 'kappa_information', 'prop_leading']

        BaseModels.__init__(self, parameter_names)

        self.duration = int(duration*fps)
        self.n_replica = int(n_replica)
        self.fps = int(fps)

        self.res_orient = None
        self.res_attachments = None

        self.rd_orientation = None
        self.rd_info = None
        self.rd_attachment = None
        self.rd_inside = None

        self.init(new)

        x = np.arange(0, self.duration + 1, 1/fps)

        self.p_outside = Fits.exp_fct(x, -.034, -.403, 0.498)/self.fps
        self.p_inside = Fits.exp_fct(x, -.025, .386, 0.135)/self.fps

    def get_p_inside(self):
        return self.p_inside[self.t]*self.para.prop_leading

    def get_p_outside(self):
        return self.p_outside[self.t]*self.para.prop_leading

    def init(self, new):
        if new is True or not self.exp.is_name_existing(self.name_orient):
            index = \
                [(id_exp, int(t*100/self.fps)) for id_exp in range(1, self.n_replica+1) for t in range(self.duration+1)]

            self.exp.add_new_empty_dataset(name=self.name_orient, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO simple model',
                                           description='UO simple model with parameter c_inside, c_outside, '
                                                       'outside attachment probability = p_outside,'
                                                       'inside attachment probability = p_inside,'
                                                       'orientation variance = 1/kappa_orientation, '
                                                       'orientation information = 1/kappa_information')

            self.exp.add_new_empty_dataset(
                name=self.name_orient+'_attachments', index_names=[id_exp_name, self.time_name],
                column_names=[], index_values=index, category='Models', label='UO simple model',
                description='UO simple model with parameter c_inside, c_outside, '
                            'outside attachment probability = p_outside,'
                            'inside attachment probability = p_inside,'
                            'orientation variance = 1/kappa_orientation, '
                            'orientation information = 1/kappa_information')

            self.exp.change_df(self.name_orient+'_attachments',
                               self.exp.get_df(self.name_orient+'_attachments').astype(int))
        else:

            self.exp.load(self.name_orient)
            self.n_replica = max(self.exp.get_index(self.name_orient).get_level_values(id_exp_name))
            self.duration = max(self.exp.get_index(self.name_orient).get_level_values(self.time_name))

    def run(self, para_value):

        self.para.change_parameter_values(para_value)
        self.name_column = str(self.para.get_parameter_tuple())

        if self.para.kappa_orientation == np.inf:
            self.rd_orientation = np.zeros((self.n_replica, self.duration+1))
        else:
            self.rd_orientation = np.random.vonmises(
                mu=0, kappa=self.para.kappa_orientation*self.fps, size=(self.n_replica, self.duration+1))

        self.rd_info = np.random.vonmises(
            mu=0, kappa=self.para.kappa_information, size=(self.n_replica, self.duration+1))
        self.rd_inside = np.random.uniform(low=-np.pi, high=np.pi, size=(self.n_replica, self.duration+1))
        self.rd_attachment = np.random.uniform(size=(self.n_replica, self.duration+1))

        self.res_orient = np.zeros((self.n_replica, self.duration+1))
        self.res_attachments = np.zeros((self.n_replica, self.duration+1), dtype=int)
        print(self.name_column)

        for self.id_exp in range(self.n_replica):
            self.orientation = np.around(rd.uniform(-np.pi, np.pi), 6)
            self.res_orient[self.id_exp, 0] = self.orientation
            self.t = 0

            for t in range(1, self.duration + 1):
                self.step()

        self.exp.get_df(self.name_orient)[self.name_column] = np.around(self.res_orient.ravel(), 6)
        self.exp.get_df(self.name_orient+'_attachments')[self.name_column] = self.res_attachments.ravel()

    def step(self):
        self.t += 1

        u = self.rd_attachment[self.id_exp, self.t]
        p_outside = self.get_p_outside()
        p_inside = self.get_p_inside()

        if u < p_outside:
            theta_ant = self.rd_info[self.id_exp, self.t]
            dtheta = Geo.angle_distance(self.para.c_outside*theta_ant, self.para.c_outside*self.orientation)
            self.orientation = Geo.angle_sum(self.orientation, dtheta)
            self.res_attachments[self.id_exp, self.t] = 1

        elif p_outside < u < p_outside+p_inside:
            theta_ant = self.rd_inside[self.id_exp, self.t]
            dtheta = Geo.angle_distance(self.para.c_inside*theta_ant, self.para.c_inside*self.orientation)
            self.orientation = Geo.angle_sum(self.orientation, dtheta)
            self.res_attachments[self.id_exp, self.t] = 2

        else:
            rho = self.rd_orientation[self.id_exp, self.t]
            self.orientation = Geo.angle_sum(self.orientation, rho)

        self.res_orient[self.id_exp, self.t] = self.orientation

    def write(self):
        self.exp.write(self.name_orient)
        self.exp.write(self.name_orient+'_attachments')
