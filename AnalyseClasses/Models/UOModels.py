import numpy as np
import random as rd
import Tools.MiscellaneousTools.Geometry as Geo

from DataStructure.VariableNames import id_exp_name
from ExperimentGroups import ExperimentGroups


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
        # if self.para.p_attachment == 'power':
        #     p_attachment = 1 / ((self.t - self.last_attachment_time) + 1)
        # else:
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


class RandGaussian:
    def __init__(self, var, m=0):
        self.rd_lg = 10000
        self.var = var
        self.m = m

        self.rd_idx = None
        self.ii = None
        self.init_idx_rd()

    def init_idx_rd(self):
        self.rd_idx = np.random.normal(self.m, np.sqrt(self.var), self.rd_lg)
        self.ii = 0

    def get_rd(self):
        r = self.rd_idx[self.ii]
        self.ii += 1
        if self.ii >= self.rd_lg:
            self.init_idx_rd()
        return r


class RandUniform:
    def __init__(self, a=0, b=1):
        self.rd_lg = 10000
        self.a = a
        self.b = b

        self.rd_idx = None
        self.ii = None
        self.init_idx_rd()

    def init_idx_rd(self):
        self.rd_idx = (self.b-self.a)*np.random.random(self.rd_lg)+self.a
        self.ii = 0

    def get_rd(self):
        r = self.rd_idx[self.ii]
        self.ii += 1
        if self.ii >= self.rd_lg:
            self.init_idx_rd()
        return r


class UOSimpleModel(BaseModels):
    def __init__(self, root, group, duration=300, n_replica=500, new=False):

        self.exp = ExperimentGroups(root, group)
        self.name_orient = 'UOSimpleModel'
        self.name_attachments = 'UOSimpleModelAttachments'
        parameter_names = ['c', 'p_attachment', 'var_orientation', 'var_information']

        BaseModels.__init__(self, parameter_names)

        self.duration = duration
        self.n_replica = n_replica

        self.res_orient = None
        self.res_attachments = None

        self.init(new)

    def init(self, new):
        if new is True or not self.exp.is_name_existing(self.name_orient):
            index = [(id_exp, t*100) for id_exp in range(1, self.n_replica+1) for t in range(self.duration+1)]

            self.exp.add_new_empty_dataset(name=self.name_orient, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO simple model',
                                           description='UO simple model with parameter c, '
                                                       'attachment probability = p_attachment,'
                                                       'orientation variance = var_orientation, '
                                                       'orientation information = var_information')

            self.exp.add_new_empty_dataset(name=self.name_attachments, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO simple model',
                                           description='Attachment instants of UO simple model with parameter c, '
                                                       'attachment probability = p_attachment,'
                                                       'orientation variance = var_orientation, '
                                                       'orientation information = var_information')
        else:

            self.exp.load(self.name_orient)
            self.exp.load(self.name_attachments)
            self.n_replica = max(self.exp.get_index(self.name_orient).get_level_values(id_exp_name))
            self.duration = max(self.exp.get_index(self.name_orient).get_level_values(self.time_name))

    def run(self, para_value):

        self.para.change_parameter_values(para_value)
        self.name_column = str(self.para.get_parameter_tuple())

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

        if u < p_attachment:
            theta_ant = rd.normalvariate(0, np.sqrt(self.para.var_information))

            dtheta = theta_ant - self.orientation
            self.orientation += self.para.c * dtheta

            self.last_attachment_time = self.t

            self.res_attachments.append(1)
        else:
            self.res_attachments.append(0)

            rho = rd.normalvariate(0, np.sqrt(self.para.var_orientation))
            self.orientation += rho

        self.res_orient.append(np.around(self.orientation, 3))

    def write(self):
        self.exp.write(self.name_orient)
        self.exp.write(self.name_attachments)


class UOCSimpleModel(BaseModels):
    def __init__(self, root, group, duration=300, n_replica=500, new=False):

        self.exp = ExperimentGroups(root, group)
        self.name_orient = 'UOCSimpleModel'
        self.name_attachments = 'UOCSimpleModelAttachments'
        parameter_names = ['p_attachment', 'var_orientation', 'var_information']

        BaseModels.__init__(self, parameter_names)

        self.duration = duration
        self.n_replica = n_replica

        self.res_orient = None
        self.res_attachments = None

        self.init(new)

        self.rand_orientation = None
        self.rand_info = None
        self.rand_init = RandUniform(-np.pi, np.pi)
        self.rand_01 = RandUniform()

    def get_c(self):
        c1 = 0.95
        t0 = 60.
        if self.t < t0:
            c0 = 0.31
            return (c1 - c0) / t0 * self.t + c0
        else:
            return c1

    def init(self, new):
        if new is True or not self.exp.is_name_existing(self.name_orient):
            index = [(id_exp, t*100) for id_exp in range(1, self.n_replica+1) for t in range(self.duration+1)]

            self.exp.add_new_empty_dataset(name=self.name_orient, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO simple model with random c',
                                           description='UO simple model with random c and parameters '
                                                       'c, variance of c = var_c,'
                                                       'attachment probability = p_attachment,'
                                                       'orientation variance = var_orientation, '
                                                       'orientation information = var_information')

            self.exp.add_new_empty_dataset(name=self.name_attachments, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO simple model',
                                           description='Attachment instants of UO simple model with random c'
                                                       ' and parameters '
                                                       'c, variance of c = var_c,'
                                                       'attachment probability = p_attachment,'
                                                       'orientation variance = var_orientation, '
                                                       'orientation information = var_information')
        else:

            self.exp.load(self.name_orient)
            self.exp.load(self.name_attachments)
            self.n_replica = max(self.exp.get_index(self.name_orient).get_level_values(id_exp_name))
            self.duration = max(self.exp.get_index(self.name_orient).get_level_values(self.time_name))

    def run(self, para_value):

        self.para.change_parameter_values(para_value)
        self.name_column = str(self.para.get_parameter_tuple())

        self.rand_orientation = RandGaussian(self.para.var_orientation)
        self.rand_info = RandGaussian(self.para.var_information)

        self.res_orient = []
        self.res_attachments = []
        print(self.name_column)

        for id_exp in range(1, self.n_replica + 1):
            self.id_exp = id_exp
            self.orientation = np.around(self.rand_init.get_rd(), 3)
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

        u = self.rand_01.get_rd()
        p_attachment = self.get_p_attachment()

        if u < p_attachment:
            theta_ant = self.rand_info.get_rd()

            dtheta = theta_ant - self.orientation
            c = self.get_c()
            self.orientation += c * dtheta

            self.last_attachment_time = self.t

            self.res_attachments.append(1)
        else:
            self.res_attachments.append(0)

            rho = self.rand_orientation.get_rd()
            self.orientation += rho

        self.res_orient.append(np.around(self.orientation, 3))

    def write(self):
        self.exp.write(self.name_orient)
        self.exp.write(self.name_attachments)


class UORandomcSimpleModel(BaseModels):
    def __init__(self, root, group, duration=300, n_replica=500, new=False):

        self.exp = ExperimentGroups(root, group)
        self.name_orient = 'UORandomcSimpleModel'
        self.name_attachments = 'UORandomcSimpleModelAttachments'
        parameter_names = ['c', 'var_c', 'p_attachment', 'var_orientation', 'var_information']

        BaseModels.__init__(self, parameter_names)

        self.duration = duration
        self.n_replica = n_replica

        self.res_orient = None
        self.res_attachments = None

        self.init(new)

        self.rand_orientation = None
        self.rand_info = None
        self.rand_c = None
        self.rand_init = RandUniform(-np.pi, np.pi)
        self.rand_01 = RandUniform()

    def init(self, new):
        if new is True or not self.exp.is_name_existing(self.name_orient):
            index = [(id_exp, t*100) for id_exp in range(1, self.n_replica+1) for t in range(self.duration+1)]

            self.exp.add_new_empty_dataset(name=self.name_orient, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO simple model with random c',
                                           description='UO simple model with random c and parameters '
                                                       'c, variance of c = var_c,'
                                                       'attachment probability = p_attachment,'
                                                       'orientation variance = var_orientation, '
                                                       'orientation information = var_information')

            self.exp.add_new_empty_dataset(name=self.name_attachments, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO simple model',
                                           description='Attachment instants of UO simple model with random c'
                                                       ' and parameters '
                                                       'c, variance of c = var_c,'
                                                       'attachment probability = p_attachment,'
                                                       'orientation variance = var_orientation, '
                                                       'orientation information = var_information')
        else:

            self.exp.load(self.name_orient)
            self.exp.load(self.name_attachments)
            self.n_replica = max(self.exp.get_index(self.name_orient).get_level_values(id_exp_name))
            self.duration = max(self.exp.get_index(self.name_orient).get_level_values(self.time_name))

    def run(self, para_value):

        self.para.change_parameter_values(para_value)
        self.name_column = str(self.para.get_parameter_tuple())

        self.rand_orientation = RandGaussian(self.para.var_orientation)
        self.rand_info = RandGaussian(self.para.var_information)
        self.rand_c = RandGaussian(m=self.para.c, var=self.para.var_c)

        self.res_orient = []
        self.res_attachments = []
        print(self.name_column)

        for id_exp in range(1, self.n_replica + 1):
            self.id_exp = id_exp
            self.orientation = np.around(self.rand_init.get_rd(), 3)
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

        u = self.rand_01.get_rd()
        p_attachment = self.get_p_attachment()

        if u < p_attachment:
            theta_ant = self.rand_info.get_rd()

            dtheta = theta_ant - self.orientation
            c = self.rand_c.get_rd()
            self.orientation += c * dtheta

            self.last_attachment_time = self.t

            self.res_attachments.append(1)
        else:
            self.res_attachments.append(0)

            rho = self.rand_orientation.get_rd()
            self.orientation += rho

        self.res_orient.append(np.around(self.orientation, 3))

    def write(self):
        self.exp.write(self.name_orient)
        self.exp.write(self.name_attachments)


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
            self.confidence = 3/np.pi**2

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
            self.confidence = self.confidence+self.c_ant
        else:
            self.confidence = 1/(self.para.var_orientation+1/self.confidence)

            rho = rd.normalvariate(0, np.sqrt(self.para.var_orientation))
            self.orientation += rho

        self.res_orientation.append(np.around(self.orientation, 3))
        self.res_confidence.append(np.around(self.confidence, 3))

    def write(self):
        self.exp.write([self.name, self.name_confidence])


class UOUniformSimpleModel(BaseModels):
    def __init__(self, root, group, duration=300, n_replica=500, new=False):

        self.exp = ExperimentGroups(root, group)
        self.name_orient = 'UOUniformSimpleModel'
        self.name_attachments = 'UOUniformSimpleModelAttachments'
        parameter_names = ['c', 'p_attachment', 'c_orientation', 'var_information']

        BaseModels.__init__(self, parameter_names)

        self.duration = duration
        self.n_replica = n_replica

        self.res_orient = None
        self.res_attachments = None

        self.init(new)

    def init(self, new):
        if new is True or not self.exp.is_name_existing(self.name_orient):
            index = [(id_exp, t*100) for id_exp in range(1, self.n_replica+1) for t in range(self.duration+1)]

            self.exp.add_new_empty_dataset(name=self.name_orient, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO Uniform simple model',
                                           description='UO Uniform simple model with parameter c, '
                                                       'attachment probability = p_attachment,'
                                                       'orientation noise = c_orientation, '
                                                       'orientation information = var_information')

            self.exp.add_new_empty_dataset(name=self.name_attachments, index_names=[id_exp_name, self.time_name],
                                           column_names=[], index_values=index, category='Models',
                                           label='UO Uniform simple model',
                                           description='Attachment instants of UO Uniform simple model'
                                                       ' with parameter c, '
                                                       'attachment probability = p_attachment,'
                                                       'orientation noise = c_orientation, '
                                                       'orientation information = var_information')
        else:

            self.exp.load(self.name_orient)
            self.exp.load(self.name_attachments)
            self.n_replica = max(self.exp.get_index(self.name_orient).get_level_values(id_exp_name))
            self.duration = max(self.exp.get_index(self.name_orient).get_level_values(self.time_name))

    def run(self, para_value):

        self.para.change_parameter_values(para_value)
        self.name_column = str(self.para.get_parameter_tuple())

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

        if u < p_attachment:
            theta_ant = rd.normalvariate(0, np.sqrt(self.para.var_information))

            dtheta = theta_ant - self.orientation
            # self.orientation += self.para.c * dtheta
            self.orientation = Geo.angle_sum(self.orientation, self.para.c * dtheta)

            self.last_attachment_time = self.t

            self.res_attachments.append(1)
        else:
            self.res_attachments.append(0)

            rho = rd.uniform(-np.pi, np.pi) * self.para.c_orientation
            # self.orientation += rho
            self.orientation = Geo.angle_sum(self.orientation, rho)

        self.res_orient.append(np.around(self.orientation, 3))

    def write(self):
        self.exp.write(self.name_orient)
        self.exp.write(self.name_attachments)
