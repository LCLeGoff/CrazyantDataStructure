import pandas as pd
import numpy as np

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_exp_name, id_ant_name, id_frame_name
from Tools.MiscellaneousTools.Geometry import angle_distance, angle_mean, angle_df, angle_distance_df
from Tools.Plotter.Plotter import Plotter


class AnalyseAntFoodRelation(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'AntFoodRelation'

    def compute_ant_density_around_food_evol(self, redo=False):
        result_name = 'ant_density_around_food_evol'
        init_frame_name = 'food_first_frame'

        hist_label = 'Evolution of ant density around the food over time'
        hist_description = 'Evolution of ant density around the food over time'

        dx = 0.25
        start_frame_intervals = np.array(np.arange(0, 3.5, dx) * 60 * 100, dtype=int)
        end_frame_intervals = np.array(start_frame_intervals + dx * 60 * 100 * 2, dtype=int)
        self.__compute_ant_density_around_food(
            result_name, init_frame_name, start_frame_intervals, end_frame_intervals, hist_label, hist_description,
            redo=redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta$ (rad)', ylabel='PDF', normed=True, label_suffix='s',
                               title='')
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def compute_ant_density_around_food_evol_first_outside_attachment(self, redo=False):
        result_name = 'ant_density_around_food_evol_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        hist_label = 'Evolution of ant density around the food over time'
        hist_description = 'Evolution of ant density around the food over time,' \
                           ' time 0 being the first outside attachment'

        dx = 0.25
        start_frame_intervals = np.arange(-1, 3., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        self.__compute_ant_density_around_food(
            result_name, init_frame_name, start_frame_intervals, end_frame_intervals, hist_label, hist_description,
            redo=redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta$ (rad)', ylabel='PDF', normed=True, label_suffix='s',
                               title='')
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def compute_outside_ant_density_around_food_evol_first_outside_attachment(self, redo=False):
        result_name = 'outside_ant_density_around_food_evol_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        hist_label = 'Evolution of outside ant density around the food over time'
        hist_description = 'Evolution of outside ant density around the food over time,' \
                           ' time 0 being the first outside attachment'

        dx = 0.25
        start_frame_intervals = np.arange(-1, 3., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        self.__compute_ant_density_around_food(
            result_name, init_frame_name, start_frame_intervals, end_frame_intervals, hist_label, hist_description,
            outside=True, redo=redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta$ (rad)', ylabel='PDF', normed=True, label_suffix='s',
                               title='')
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def compute_non_outside_ant_density_around_food_evol_first_outside_attachment(self, redo=False):
        result_name = 'non_outside_ant_density_around_food_evol_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        hist_label = 'Evolution of inside ant density around the food over time'
        hist_description = 'Evolution of inside ant density around the food over time,' \
                           ' time 0 being the first outside attachment'

        dx = 0.25
        start_frame_intervals = np.arange(-1, 3., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        self.__compute_ant_density_around_food(
            result_name, init_frame_name, start_frame_intervals, end_frame_intervals, hist_label, hist_description,
            outside=False, redo=redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta$ (rad)', ylabel='PDF', normed=True, label_suffix='s',
                               title='')
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def compute_slowing_ant_density_around_food_evol_first_outside_attachment(self, redo=False):
        result_name = 'slowing_ant_density_around_food_evol_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        hist_label = 'Evolution of slowing ant density around the food over time'
        hist_description = 'Evolution of slowing ant density around the food over time,' \
                           ' means density of the position sof ants slowing down, i.e. which may marking,' \
                           ' time 0 being the first outside attachment'

        dx = 0.25
        start_frame_intervals = np.arange(-1, 3., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        self.__compute_ant_density_around_food(
            result_name, init_frame_name, start_frame_intervals, end_frame_intervals, hist_label, hist_description,
            speed=True, redo=redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta$ (rad)', ylabel='PDF', normed=True, label_suffix='s',
                               title='')
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def compute_slowing_outside_ant_density_around_food_evol_first_outside_attachment(self, redo=False):
        result_name = 'slowing_outside_ant_density_around_food_evol_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        hist_label = 'Evolution of slowing outside ant density around the food over time'
        hist_description = 'Evolution of slowing outside ant density around the food over time,' \
                           ' means density of the position sof ants slowing down, i.e. which may marking,' \
                           ' time 0 being the first outside attachment'

        dx = 0.25
        start_frame_intervals = np.arange(-1, 3., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        self.__compute_ant_density_around_food(
            result_name, init_frame_name, start_frame_intervals, end_frame_intervals, hist_label, hist_description,
            speed=True, outside=True, redo=redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta$ (rad)', ylabel='PDF', normed=True, label_suffix='s',
                               title='')
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def compute_slowing_non_outside_ant_density_around_food_evol_first_outside_attachment(self, redo=False):
        result_name = 'slowing_non_outside_ant_density_around_food_evol_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        hist_label = 'Evolution of slowing inside ant density around the food over time'
        hist_description = 'Evolution of slowing inside ant density around the food over time,' \
                           ' means density of the position sof ants slowing down, i.e. which may marking,' \
                           ' time 0 being the first outside attachment'

        dx = 0.25
        start_frame_intervals = np.arange(-1, 3., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        self.__compute_ant_density_around_food(
            result_name, init_frame_name, start_frame_intervals, end_frame_intervals, hist_label, hist_description,
            speed=True, outside=False, redo=redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta$ (rad)', ylabel='PDF', normed=True, label_suffix='s',
                               title='')
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def __compute_ant_density_around_food(
            self, result_name, init_frame_name, start_frame_intervals, end_frame_intervals,
            hist_label, hist_description, outside=None, speed=False, redo=False):

        dist = 60
        speed_min = 1.

        dtheta = 2*np.pi / 25.
        bins = np.arange(0, np.pi + dtheta, dtheta)

        if redo:
            df_dist = self._get_times_for_ant_density(
                init_frame_name=init_frame_name, dist=dist, speed_min=speed_min, outside=outside, speed=speed)

            ant_food_phi_name = 'ant_food_phi'
            exit_angle_name = 'food_exit_angle'
            dist2food_name = 'distance2food'

            df_exit_angle = self.reindexing_exp_frame_indexed_by_exp_ant_frame_indexed(
                exit_angle_name, dist2food_name, column_names=self.exp.get_columns(ant_food_phi_name))

            df = self.exp.get_df(ant_food_phi_name).where(df_dist[dist2food_name], np.nan)
            arr = angle_distance(df_exit_angle, df)
            df[:] = np.abs(arr)
            self.exp.change_df(ant_food_phi_name, df.dropna())

            self.change_first_frame(ant_food_phi_name, init_frame_name)

            self.exp.hist1d_evolution(name_to_hist=ant_food_phi_name, start_frame_intervals=start_frame_intervals,
                                      end_frame_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category,
                                      label=hist_label, description=hist_description)

            self.exp.remove_object(ant_food_phi_name)
            self.exp.remove_object(dist2food_name)
            self.exp.remove_object(exit_angle_name)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

    def _get_times_for_ant_density(self, init_frame_name, dist, speed_min=None, outside=None, speed=None):

        exit_angle_name = 'food_exit_angle'
        ant_food_phi_name = 'ant_food_phi'
        dist2food_name = 'distance2food'
        dist2border_name = 'food_border_distance'
        name_radius = 'food_radius'

        self.exp.load([dist2food_name, dist2border_name, exit_angle_name,
                       ant_food_phi_name, init_frame_name, name_radius])

        df_radius = self.reindexing_exp_indexed_by_exp_ant_frame_indexed(name_radius, dist2food_name)
        df_border_dist = self.reindexing_exp_frame_indexed_by_exp_ant_frame_indexed(
            dist2border_name, dist2food_name, column_names=self.exp.get_columns(dist2food_name))

        df_dist = self.exp.get_df(dist2food_name).copy()
        df_dist2 = df_dist < dist
        df_dist2 &= df_radius * 1.1 < df_dist
        df_dist2 &= df_border_dist > dist

        if outside is not None:
            outside_name = 'from_outside'
            self.exp.load(outside_name)
            df_from_outside = self.__reindexing_exp_ant_indexed_by_exp_ant_frame_indexed(
                outside_name, dist2food_name, column_names=self.exp.get_columns(dist2food_name))
            if outside is True:
                df_dist2 &= df_from_outside.astype(bool)
            else:
                df_dist2 &= (1 - df_from_outside).astype(bool)

        if speed is True:
            speed_name = 'speed'
            self.exp.load(speed_name)

            df_speed = self.exp.get_df(speed_name) < speed_min
            df_speed.columns = df_dist2.columns
            df_speed = df_speed.reindex(df_dist2.index)
            df_dist2 &= df_speed

        return df_dist2

    def loop_density_around_food_evol(self, redo=True):
        result_name = 'loop_density_around_food_evol'
        init_frame_name = 'food_first_frame'

        hist_label = 'Evolution of loop density around the food'
        hist_description = 'Evolution of the mean of ant positions (radial coordinate),' \
                           ' while the ant is looping around the food'

        dx = 0.5
        start_frame_intervals = np.arange(0, 3., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        self._get_loop_density(result_name, init_frame_name, start_frame_intervals, end_frame_intervals, hist_label,
                               hist_description, outside=None, speed=False, redo=redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta$ (rad)', ylabel='PDF', normed=True, label_suffix='s',
                               title='')
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def loop_density_around_food_evol_first_outside_attachment(self, redo=True):
        result_name = 'loop_density_around_food_evol_first_outside_attachment'
        init_frame_name = 'first_attachment_time_of_outside_ant'

        hist_label = 'Evolution of loop density around the food'
        hist_description = 'Evolution of the mean of ant positions (radial coordinate),' \
                           ' while the ant is looping around the food'

        dx = 0.5
        start_frame_intervals = np.arange(-1, 3., dx)*60*100
        end_frame_intervals = start_frame_intervals + dx*60*100*2

        self._get_loop_density(result_name, init_frame_name, start_frame_intervals, end_frame_intervals, hist_label,
                               hist_description, outside=None, speed=False, redo=redo)

        plotter = Plotter(root=self.exp.root, obj=self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(xlabel=r'$\theta$ (rad)', ylabel='PDF', normed=True, label_suffix='s',
                               title='')
        plotter.draw_legend(ax=ax, ncol=2)
        plotter.save(fig)

    def _get_loop_density(self, result_name, init_frame_name, start_frame_intervals, end_frame_intervals, hist_label,
                          hist_description, outside, speed, redo):
        speed_min = 1.
        dist = 60
        dtheta = np.pi / 25.
        bins = np.arange(0, np.pi + dtheta, dtheta)
        if redo:
            df_dist = self._get_times_for_ant_density(
                init_frame_name=init_frame_name, dist=dist, speed_min=speed_min, outside=outside, speed=speed)

            ant_food_phi_name = 'ant_food_phi'
            exit_angle_name = 'food_exit_angle'
            dist2food_name = 'distance2food'

            df_exit_angle = self.reindexing_exp_frame_indexed_by_exp_ant_frame_indexed(
                exit_angle_name, dist2food_name, column_names=self.exp.get_columns(ant_food_phi_name))

            df_phi = self.exp.get_df(ant_food_phi_name).where(df_dist[dist2food_name], np.nan)
            arr = angle_distance(df_exit_angle, df_phi)
            df_phi[:] = np.abs(arr)
            df_phi = df_phi.dropna()
            self.exp.change_df(ant_food_phi_name, df_phi)

            not_carrying_interval_name = 'not_carrying_intervals'
            self.exp.load([not_carrying_interval_name, 'fps'])
            mean_pos_name = 'means'
            self.exp.add_copy(old_name=not_carrying_interval_name, new_name=mean_pos_name, replace=True)
            self.exp.get_df(mean_pos_name)[:] = np.nan

            def add_loop_index4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                print(id_exp)
                fps = self.exp.get_value('fps', id_exp)
                intervals = self.exp.get_df(not_carrying_interval_name).loc[id_exp, :, :].reset_index().values
                intervals[:, -1] *= fps
                intervals[:, -1] += intervals[:, -2]

                for id_exp, id_ant, frame0, frame1 in intervals.astype(int):
                    temp_df = df.loc[id_exp, id_ant, frame0:frame1]
                    if len(temp_df) != 0:
                        angle_tab = temp_df.values
                        m = round(angle_mean(angle_tab[~np.isnan(angle_tab)]), 6)
                        self.exp.get_df(mean_pos_name).loc[id_exp, id_ant, frame0] = m

                return df

            self.exp.groupby(ant_food_phi_name, id_exp_name, add_loop_index4each_group)

            self.change_first_frame(mean_pos_name, init_frame_name)

            self.exp.hist1d_evolution(name_to_hist=mean_pos_name, start_frame_intervals=start_frame_intervals,
                                      end_frame_intervals=end_frame_intervals, bins=bins,
                                      result_name=result_name, category=self.category,
                                      label=hist_label, description=hist_description)

            self.exp.remove_object(ant_food_phi_name)
            self.exp.remove_object(dist2food_name)
            self.exp.remove_object(exit_angle_name)
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

    def reindexing_exp_indexed_by_exp_ant_frame_indexed(self, name2reindex, name_of_index):
        df = self.exp.get_df(name_of_index).copy()
        df[:] = np.nan
        for id_exp in self.exp.get_index(name2reindex):
            df.loc[id_exp, :, :] = self.exp.get_value(name2reindex, id_exp)

        return df

    def __reindexing_exp_ant_indexed_by_exp_ant_frame_indexed(self, name_to_reindex, name2, column_names=None):
        if column_names is None:
            column_names = self.exp.get_columns(name_to_reindex)

        id_exps = self.exp.get_df(name2).index.get_level_values(id_exp_name)
        id_ants = self.exp.get_df(name2).index.get_level_values(id_ant_name)
        frames = self.exp.get_df(name2).index.get_level_values(id_frame_name)
        idxs = pd.MultiIndex.from_tuples(list(zip(id_exps, id_ants)), names=[id_exp_name, id_ant_name])

        df = self.exp.get_df(name_to_reindex).copy()
        df = df.reindex(idxs)
        df[id_frame_name] = frames
        df.reset_index(inplace=True)
        df.columns = [id_exp_name, id_ant_name]+column_names+[id_frame_name]
        df.set_index([id_exp_name, id_ant_name, id_frame_name], inplace=True)

        return df

    def compute_foodVelocity_foodAntAttachmentVector_angle(self):
        vector_name = 'foodVelocity_foodAntAttachmentVector_angle'

        ant_xy_name = 'ant_xy'
        self.exp.load_as_2d('mm10_x', 'mm10_y', ant_xy_name, 'x', 'y', replace=True)

        food_speed_name = 'food_velocity'
        self.exp.load_as_2d(food_speed_name+'_x', food_speed_name+'_y', food_speed_name, 'x', 'y', replace=True)
        df_food_speed = self.reindexing_exp_frame_indexed_by_exp_ant_frame_indexed(food_speed_name, ant_xy_name)

        food_xy_name = 'mm10_food_xy'
        self.exp.load_as_2d(food_xy_name[:-2]+'x', food_xy_name[:-2]+'y', food_xy_name, 'x', 'y', replace=True)
        df_food_xy = self.reindexing_exp_frame_indexed_by_exp_ant_frame_indexed(food_xy_name, ant_xy_name)

        df_ant_food_vector = self.exp.get_df(ant_xy_name)-df_food_xy

        df_angle = angle_df(df_ant_food_vector, df_food_speed)

        label = 'Food velocity, food-ant attachment position vector angle (rad)'
        description = 'Angle between the food velocity and the food-ant attachment position vector (rad)'
        self.exp.add_copy(
            old_name='mm10_x', new_name=vector_name, category=self.category, label=label, description=description)
        self.exp.change_df(vector_name, df_angle)

        self.exp.write(vector_name)

    def compute_mm10_foodVelocity_foodAntAttachmentVector_angle(self):
        name = 'foodVelocity_foodAntAttachmentVector_angle'
        window = 10

        self.exp.load(name)
        result_name = self.exp.rolling_mean(name_to_average=name, window=window, category=self.category, is_angle=True)
        self.exp.write(result_name)

    def compute_mm1s_foodVelocity_foodAntAttachmentVector_angle(self):
        name = 'foodVelocity_foodAntAttachmentVector_angle'
        window = 100

        self.exp.load(name)
        result_name = self.exp.rolling_mean(
            name_to_average=name, result_name='mm1s_'+name,
            window=window, category=self.category, is_angle=True)
        self.exp.write(result_name)

    def compute_foodVelocity_AntBodyOrientation_angle(self):
        vector_name = 'foodVelocity_AntBodyOrientation_angle'

        ant_body_name = 'mm10_orientation'
        food_orientation_name = 'food_velocity_phi'
        self.exp.load([ant_body_name, food_orientation_name])

        df_ant_body = self.exp.get_df(ant_body_name)
        df_food_orientation = self.reindexing_exp_frame_indexed_by_exp_ant_frame_indexed(
            food_orientation_name, ant_body_name)

        df_angle = angle_distance_df(df_ant_body, df_food_orientation)

        label = 'Food orientation - ant body orientation vector angle (rad)'
        description = 'Angle between the food orientation and the ant body orientation vector (rad)'
        self.exp.add_copy(
            old_name=ant_body_name, new_name=vector_name, category=self.category, label=label, description=description)
        self.exp.change_df(vector_name, df_angle)

        self.exp.write(vector_name)

    def compute_foodRotation_AntBodyOrientation_angle(self):
        vector_name = 'foodRotation_AntBodyOrientation_angle'

        ant_body_name = 'mm10_orientation'
        food_rotation_name = 'food_rotation'
        self.exp.load([ant_body_name, food_rotation_name])

        df_ant_body = self.exp.get_df(ant_body_name)
        df_food_rotation = self.reindexing_exp_frame_indexed_by_exp_ant_frame_indexed(
            food_rotation_name, ant_body_name)

        df_angle = angle_distance_df(df_ant_body, df_food_rotation)

        label = 'Food rotation - ant body orientation vector angle (rad)'
        description = 'Angle between the food rotation and the ant body orientation vector (rad)'
        self.exp.add_copy(
            old_name=ant_body_name, new_name=vector_name, category=self.category, label=label, description=description)
        self.exp.change_df(vector_name, df_angle)

        self.exp.write(vector_name)

    def compute_foodVelocity_foodAntVector_angle(self):
        vector_name = 'foodVelocity_foodAntVector_angle'

        ant_xy_name = 'ant_xy'
        self.exp.load_as_2d('mm10_x', 'mm10_y', ant_xy_name, 'x', 'y', replace=True)

        food_speed_name = 'food_velocity'
        self.exp.load_as_2d(food_speed_name+'_x', food_speed_name+'_y', food_speed_name, 'x', 'y', replace=True)
        df_food_speed = self.reindexing_exp_frame_indexed_by_exp_ant_frame_indexed(food_speed_name, ant_xy_name)

        food_xy_name = 'mm10_food_xy'
        self.exp.load_as_2d(food_xy_name[:-2]+'x', food_xy_name[:-2]+'y', food_xy_name, 'x', 'y', replace=True)
        df_food_xy = self.reindexing_exp_frame_indexed_by_exp_ant_frame_indexed(food_xy_name, ant_xy_name)

        df_ant_food_vector = self.exp.get_df(ant_xy_name)-df_food_xy

        df_angle = angle_df(df_ant_food_vector, df_food_speed)

        label = 'Food velocity, food-ant vector angle (rad)'
        description = 'Angle between the food velocity and the food-ant vector (rad)'
        self.exp.add_copy(
            old_name='mm10_x', new_name=vector_name, category=self.category, label=label, description=description)
        self.exp.change_df(vector_name, df_angle)

        self.exp.write(vector_name)

    def compute_foodVelocity_foodAntVector_angle_around_attachments(self, redo=False):
        name = 'foodVelocity_foodAntVector_angle'
        attachment_name = 'carrying_intervals'
        result_name = 'foodVelocity_foodAntVector_angle_around_attachments'

        label = 'Distribution of food velocity, food-ant vector angle around attachments (rad)'
        description = 'Distribution of the angle between the food velocity and the food-ant vector (rad),' \
                      '0, 1, ...8 seconds after an attachment'

        self._get_vector_angle_around_attachment(name, attachment_name, result_name, label, description, redo)

    def compute_foodVelocity_foodAntVector_angle_around_outside_attachments(self, redo=False):
        name = 'foodVelocity_foodAntVector_angle'
        attachment_name = 'outside_ant_carrying_intervals'
        result_name = 'foodVelocity_foodAntVector_angle_around_outside_attachments'

        label = 'Distribution of food velocity, food-ant vector angle around outside attachments (rad)'
        description = 'Distribution of the angle between the food velocity and the food-ant vector (rad),' \
                      '0, 1, ...8 seconds after an outside attachment'

        self._get_vector_angle_around_attachment(name, attachment_name, result_name, label, description, redo)

    def _get_vector_angle_around_attachment(self, name, attachment_name, result_name, label, description, redo):
        dtheta = np.pi / 10.
        bin_thetas = np.around(np.arange(-np.pi - dtheta / 2., np.pi + dtheta / 2., dtheta), 2)
        bin_thetas2 = (bin_thetas[1:] + bin_thetas[:-1]) / 2.

        if redo:
            self.exp.load([name, attachment_name, 'fps'])
            res = np.full((len(self.exp.get_df(attachment_name)), 9), np.nan)
            list_iter = np.zeros(9, dtype=int)

            def get_angle4each_group(df: pd.DataFrame):
                id_exp = df.index.get_level_values(id_exp_name)[0]
                print(id_exp)
                fps = self.exp.get_value('fps', id_exp)
                tab_attach = self.exp.get_df(attachment_name).loc[id_exp, :].reset_index().values

                for id_ant, frame0, time in tab_attach:
                    time = min(time, 8)
                    frame1 = frame0 + time * fps

                    f0 = int(frame0)
                    j = 0
                    while f0 < frame1:
                        f1 = f0 + int(fps)

                        theta = np.nanmean(df.loc[id_exp, id_ant, f0:f1].values)
                        res[list_iter[j], j] = np.around(theta, 6)

                        list_iter[j] += 1
                        j += 1
                        f0 = f1

            self.exp.groupby(name, id_exp_name, get_angle4each_group)
            self.exp.add_new_empty_dataset(name=result_name, index_names='bin_thetas', column_names=range(9),
                                           index_values=bin_thetas2, category=self.category, label=label,
                                           description=description)
            for i in range(9):
                thetas = res[:list_iter[i], i]
                y, x = np.histogram(thetas, bins=bin_thetas)
                self.exp.get_df(result_name)[i] = y
            self.exp.write(result_name)
        else:
            self.exp.load(result_name)

        plotter = Plotter(self.exp.root, self.exp.get_data_object(result_name))
        fig, ax = plotter.plot(normed=True)
        plotter.save(fig)

    def compute_angle_body_food_orientation(self):
        name = 'angle_body_food'

        name_food_orientation = 'food_velocity_phi'
        name_ant_orientation = 'mm10_orientation'
        self.exp.load([name_ant_orientation, name_food_orientation])

        df_food_orientation = self.reindexing_exp_frame_indexed_by_exp_ant_frame_indexed(
            name_food_orientation, name_ant_orientation)

        df_ant_orientation = self.exp.get_df(name_ant_orientation)

        df_angle_body = angle_distance_df(df_ant_orientation, df_food_orientation)

        self.exp.add_new1d_from_df(df=df_angle_body, name=name, object_type='TimeSeries1d', category=self.category,
                                   label='body-food orientation angle',
                                   description='Angle between the food orientation and the body orientation')

        self.exp.write(name)
