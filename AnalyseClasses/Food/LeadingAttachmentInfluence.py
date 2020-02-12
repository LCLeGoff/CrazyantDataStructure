import numpy as np
import pandas as pd
import Tools.MiscellaneousTools.Geometry as Geo

from matplotlib import pylab as plt

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.Plotter.Plotter import Plotter


class AnalyseLeadingAttachmentInfluence(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'LeadingAttachmentInfluence'

    def compute_leading_attachment_influence(self):
        food_orientation_name = 'mm10_food_velocity_phi'
        leading_attachment_name = 'leading_attachments'
        food_speed_name = 'mm10_food_speed'
        self.exp.load([food_orientation_name, leading_attachment_name, food_speed_name])

        id_exp = 30

        df_orient = self.exp.get_df(food_orientation_name).loc[id_exp, :].dropna()
        orient_tab = df_orient.values
        orient_tab = Geo.angle_distance(orient_tab[1:], orient_tab[:-1])
        orient_tab = np.cumsum(orient_tab)

        df_orient.iloc[1:] = np.c_[orient_tab]
        df_orient.iloc[1:] += df_orient.iloc[0]

        df_speed = self.exp.get_df(food_speed_name).loc[id_exp, :]
        df_attach = self.exp.get_df(leading_attachment_name).loc[id_exp, :]

        index = df_orient.index.copy()
        df_orient2 = df_orient.copy()
        df_orient2.index -= 10
        df_orient.index += 10

        df_dorient = Geo.angle_distance_df(df_orient % 2*np.pi, df_orient2 % 2*np.pi)
        df_dorient = df_dorient.reindex(index)

        df_dorient2 = df_dorient[df_speed[food_speed_name] > 1].dropna()
        df_dorient2 = df_dorient2[df_dorient2.abs() > 1.1].dropna()
        t = np.array(df_dorient2.index[1:])-np.array(df_dorient2.index[:-1])
        t = t[t > 1]




