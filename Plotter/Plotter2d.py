from matplotlib import pyplot as plt
import numpy as np
from Plotter.ColorObject import ColorObject
from Plotter.BasePlotters import BasePlotters
from Plotter.FeatureArguments import ArgumentsTools, ArenaFeatureArguments, FoodFeatureArguments, GateFeatureArguments, \
	LineFeatureArguments


class Plotter2d(BasePlotters):
	def __init__(
			self, obj, arena_length=420., arena_width=297, circular_arena_radius=200., gate_length=90.,
			food_radius=5., food_location=None, **kwargs):

		BasePlotters.__init__(self, obj)

		self.arg_tools = ArgumentsTools(self)

		self.arg_tools.add_arguments('line', LineFeatureArguments(), ls='', marker='o', c='w', alpha=1)
		self.arg_tools.change_arg_value('line', kwargs)
		self.arg_tools.add_arguments('arena', ArenaFeatureArguments(), **kwargs)
		self.arg_tools.add_arguments('food', FoodFeatureArguments(), **kwargs)
		self.arg_tools.add_arguments('gate', GateFeatureArguments(), **kwargs)

		self.arena_length = arena_length
		self.arena_width = arena_width
		self.gate_length = gate_length
		self.food_radius = food_radius
		self.circular_arena_radius = circular_arena_radius
		if food_location is None:
			self.food_location = [0, 0]

	def _change_arg_values(self, names, kwargs):
		if isinstance(names, str):
			names = [names]
		for name in names:
			self.arg_tools.change_arg_value(name, kwargs)

	@staticmethod
	def make_circle(radius=5., center=None):
		if center is None:
			center = [0, 0]
		theta = np.arange(-np.pi, np.pi+0.1, 0.1)
		return radius*np.cos(theta)+center[0], radius*np.sin(theta)+center[1]

	def draw_food(self, ax, **kwargs):
		self._change_arg_values('food', kwargs)
		x, y = self.make_circle(self.food_radius, self.food_location)
		ax.plot(x, y, **self.food)

	def draw_arena(self, ax, **kwargs):
		self._change_arg_values('arena', kwargs)
		self.draw_rectangle(ax, self.food_location, self.arena_length, self.arena_width, self.arena)

	@staticmethod
	def draw_rectangle(ax, pts, length, width, plot_features):
		x, y = pts
		ax.plot([x-length/2., x-length/2.], [y-width/2., y+width/2.], **plot_features)
		ax.plot([x+length/2., x+length/2.], [y-width/2., y+width/2.], **plot_features)
		ax.plot([x-length/2., x+length/2.], [y+width/2., y+width/2.], **plot_features)
		ax.plot([x-length/2., x+length/2.], [y-width/2., y-width/2.], **plot_features)

	def draw_gate(self, ax, **kwargs):
		self._change_arg_values('gate', kwargs)
		x = self.arena_length/2.
		ax.plot([x, x], np.array([-1, 1])*self.gate_length/2., **self.gate)

	def draw_setup(self, fig, ax, **kwargs):
		self._change_arg_values(['arena', 'gate', 'food'], kwargs)
		ax.axis('equal')
		self.remove_axis(fig, ax)
		self.draw_arena(ax)
		self.draw_gate(ax)
		self.draw_food(ax)

	def display_labels(self, ax):
		ax.set_xlabel(self.obj.definition.xlabel)
		ax.set_xlabel(self.obj.definition.xlabel)

	def repartition_in_arena(self, list_id_exp=None, color_variety=None, title_prefix=None, preplot=None, **kwargs):
		list_id_exp = self._get_list_id_exp(list_id_exp)

		self._change_arg_values('line', kwargs)

		fig, ax = self.create_plot(preplot, (6.5, 5))

		if color_variety == 'exp':
			self._plot_2d_obj_per_exp(ax, list_id_exp)
		elif color_variety == 'ant':
			self._plot_2d_obj_per_ant(ax, list_id_exp)
		elif color_variety == 'ant2':
			self._plot_2d_obj_per_ant2(ax, list_id_exp)
		elif color_variety == 'frame':
			self._plot_2d_obj_per_frame(ax, list_id_exp)
		else:
			self._plot_2d_obj(ax, list_id_exp)

		self.draw_setup(fig, ax)
		self.display_title(ax, title_prefix)

		return fig, ax

	def _plot_2d_obj_for_exp_ant(self, ax, id_exp, id_ant):
		x_array = self.obj.get_x_values().loc[id_exp, id_ant, :]
		y_array = self.obj.get_y_values().loc[id_exp, id_ant, :]
		ax.plot(x_array, y_array, **self.line)

	def _plot_2d_obj_for_exp_ant_frame(self, ax, id_exp, id_ant, frame):
		x = self.obj.get_x_values().loc[id_exp, id_ant, frame]
		y = self.obj.get_y_values().loc[id_exp, id_ant, frame]
		ax.plot(x, y, **self.line)

	def _plot_2d_obj(self, ax, list_id_exp):
		id_exp_ant_list = self.obj.get_index_array_of_id_exp_ant()
		for id_exp, id_ant in id_exp_ant_list:
			if id_exp in list_id_exp:
				self._plot_2d_obj_for_exp_ant(ax, id_exp, id_ant)

	def _plot_2d_obj_per_ant(self, ax, list_id_exp):
		id_exp_ant_list = self.obj.get_index_array_of_id_exp_ant()
		col_list_for_each_exp_ant = ColorObject('cmap', self.cmap, id_exp_ant_list).colors
		for id_exp, id_ant in id_exp_ant_list:
			if id_exp in list_id_exp:
				self.line['c'] = col_list_for_each_exp_ant[(id_exp, id_ant)]
				self._plot_2d_obj_for_exp_ant(ax, id_exp, id_ant)

	def _plot_2d_obj_per_ant2(self, ax, list_id_exp):
		id_exp_ant_list = self.obj.get_index_array_of_id_exp_ant()
		col_list_for_each_exp_ant = ColorObject('cmap', self.cmap, set(id_exp_ant_list[:, 1])).colors
		for id_exp, id_ant in id_exp_ant_list:
			if id_exp in list_id_exp:
				self.line['c'] = col_list_for_each_exp_ant[id_ant]
				self._plot_2d_obj_for_exp_ant(ax, id_exp, id_ant)

	def _plot_2d_obj_per_frame(self, ax, list_id_exp):
		col_list = ColorObject('cmap', self.cmap, 101).colors

		id_exp_ant_frame_dict = self.obj.get_index_dict_of_id_exp_ant_frame()
		for id_exp in list_id_exp:
			frame_array = self.obj.get_array_of_all_frames_of_exp(id_exp)
			exp_time_length = frame_array[-1]-frame_array[0]

			for id_ant in id_exp_ant_frame_dict[id_exp]:
				for frame in id_exp_ant_frame_dict[id_exp][id_ant]:
					col_idx = int((frame-frame_array[0])/float(exp_time_length)*100)
					col = col_list[col_idx]
					self.line['c'] = col
					self._plot_2d_obj_for_exp_ant_frame(ax, id_exp, id_ant, frame)

	def _plot_2d_obj_per_exp(self, ax, list_id_exp):
		id_exp_ant_list = self.obj.get_index_array_of_id_exp_ant()
		col_list = ColorObject('cmap', self.cmap, list_id_exp).colors
		for id_exp, id_ant in id_exp_ant_list:
			if id_exp in list_id_exp:
				self.line['c'] = col_list[id_exp]
				self._plot_2d_obj_for_exp_ant(ax, id_exp, id_ant)

	def hist2d_in_arena(self, bins=100, normed=False, title_prefix=None, preplot=None, cmap=None):
		if cmap is None:
			self.cmap = cmap

		fig, ax = self.create_plot(preplot, (9, 5))
		self.display_labels(ax)
		self.display_title(ax, title_prefix)
		plt.hist2d(self.obj.get_x_values(), self.obj.get_y_values(), bins=bins, cmap=cmap, normed=normed)
		plt.colorbar()
		plt.axis('equal')
		self.draw_setup(fig, ax)

		return fig, ax

	def radial_direction_in_arena(self, color_variety=None, preplot=None, list_id_exp=None, **kwarg):

		list_id_exp = self._get_list_id_exp(list_id_exp)

		self.line['marker'] = ''
		self.line['ls'] = '-'
		self._change_arg_values('line', kwarg)

		fig, ax = self.create_plot(preplot, (9, 5))

		if color_variety == 'exp':
			self._plot_phi_direction_per_exp(ax, list_id_exp)
		elif color_variety == 'ant':
			self._plot_phi_direction_per_ant(ax, list_id_exp)
		elif color_variety == 'ant2':
			self._plot_phi_direction_per_ant2(ax, list_id_exp)
		else:
			self._plot_all_phi_direction(ax, list_id_exp)

		self.draw_setup(fig, ax)

	def _get_list_id_exp(self, list_id_exp):
		if list_id_exp is None:
			list_id_exp = self.obj.get_index_array_of_id_exp()
		return list_id_exp

	def _plot_phi_direction_per_exp(self, ax, list_id_exp):
		col_list_for_each_exp_ant = ColorObject('cmap', self.cmap, list_id_exp).colors
		df_array = self.obj.convert_df_to_array()
		for id_exp, id_ant, frame, phi in df_array:
			if id_exp in list_id_exp:
				self.line['c'] = col_list_for_each_exp_ant[id_exp]
				self._plot_phi_direction(ax, phi)

	def _plot_phi_direction_per_ant(self, ax, list_id_exp):
		id_exp_ant_list = self.obj.get_index_array_of_id_exp_ant()
		col_list_for_each_exp_ant = ColorObject('cmap', self.cmap, id_exp_ant_list).colors
		phi_array = self.obj.convert_df_to_array()
		for id_exp, id_ant, frame, phi in phi_array:
			if id_exp in list_id_exp:
				self.line['c'] = col_list_for_each_exp_ant[(id_exp, id_ant)]
				self._plot_phi_direction(ax, phi)

	def _plot_phi_direction_per_ant2(self, ax, list_id_exp):
		id_exp_ant_list = self.obj.get_index_array_of_id_exp_ant()
		col_list_for_each_exp_ant = ColorObject('cmap', self.cmap, set(id_exp_ant_list[:, 1])).colors
		phi_array = self.obj.convert_df_to_array()
		for id_exp, id_ant, frame, phi in phi_array:
			if id_exp in list_id_exp:
				self.line['c'] = col_list_for_each_exp_ant[id_ant]
				self._plot_phi_direction(ax, phi)

	def _plot_all_phi_direction(self, ax, list_id_exp):
		phi_array = self.obj.convert_df_to_array()
		for id_exp, id_ant, frame, phi in phi_array:
			if id_exp in list_id_exp:
				self._plot_phi_direction(ax, phi)

	def _plot_phi_direction(self, ax, phi):
		x = self.circular_arena_radius * np.cos(phi)
		y = self.circular_arena_radius * np.sin(phi)
		ax.plot([0, x], [0, y], **self.line)
