# from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
from cv2 import cv2
from matplotlib import pylab as plt

from AnalyseClasses.AnalyseClassDecorator import AnalyseClassDecorator
from DataStructure.VariableNames import id_frame_name, id_ant_name
from Movies.Movies import Movies
from Scripts.root import root_movie, root


class MakeMarkingMovie(AnalyseClassDecorator):
    def __init__(self, group, exp=None):
        AnalyseClassDecorator.__init__(self, group, exp)
        self.category = 'MarkingMovie'

    def movie(self, id_exp, name_movie=None):

        label_radius = 40
        cmap_name = 'jet'
        dt_max = 10

        if name_movie is None:
            name_movie = 'movie%s' % id_exp
        movie_add = '%s%s/%s/%s.avi' % (root, self.exp.group, self.category, name_movie)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        print(movie_add)

        movie = self.exp.get_movie(id_exp)

        marking_name = 'potential_marking_intervals'
        self.exp.load(marking_name)
        xy_name = 'xy'
        self.exp.load_as_2d('mm10_x', 'mm10_y', xy_name, 'x', 'y')

        marking_df = self.exp.get_df(marking_name).loc[id_exp, :, :]
        xy = self.exp.get_df(xy_name).loc[id_exp, :, :]
        xy.x, xy.y = self.exp.convert_xy_to_movie_system(id_exp, xy.x, xy.y)

        list_ants = list(set(list(marking_df.index.get_level_values(id_ant_name))))
        list_ants.sort()
        list_frames = xy.index.get_level_values(id_frame_name)
        list_frames = list(range(min(list_frames), max(list_frames)))

        df_marking_label = pd.DataFrame(index=list_frames, columns=list_ants)
        df_marking_label[:] = np.nan

        df_color = pd.DataFrame(index=list_frames, columns=list_ants)
        df_color[:] = np.nan

        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        fig.subplots_adjust(0, 0, 1, 1)
        ax.axis('equal')
        ax.invert_yaxis()
        ax.set_axis_off()

        cmap = plt.get_cmap(cmap_name)

        frame_img = movie.get_frame(min(marking_df.index.get_level_values(id_frame_name))-dt_max)
        frame_graph = ax.imshow(frame_img)

        graph_dict = dict()
        label_dict = dict()

        frame_ant_dict = {frame: [] for frame in list_frames}

        def do4each_group(df: pd.DataFrame):
            ant = df.index.get_level_values(id_ant_name)[0]
            frames = list(df.index.get_level_values(id_frame_name))
            dts = (df.values.ravel()*100).astype(int)

            color = cmap(np.random.randint(256))
            dt1 = min(dt_max, int(dts[0]/2))
            frame = frames[0]
            df_color.loc[frame - dt_max:frame + dt1, ant] = [color]
            df_marking_label.loc[frame - dt_max:frame + dt1, ant] = frame

            temp_frames = list(xy.loc[id_exp, ant, frame - dt_max:frame + dt1].index.get_level_values(id_frame_name))
            for frame in temp_frames:
                frame_ant_dict[frame].append(ant)

            for i, frame in enumerate(frames):
                color = cmap(np.random.randint(256))
                dt0 = min(dt_max, int(dts[i-1]/2))
                dt1 = min(dt_max, int(dts[i]/2))
                df_color.loc[frame - dt0:frame + dt1, ant] = [color]
                df_marking_label.loc[frame - dt0:frame + dt1, ant] = frame

                temp_frames = list(xy.loc[id_exp, ant, frame-dt_max:frame+dt1].index.get_level_values(id_frame_name))
                for temp_frame in temp_frames:
                    frame_ant_dict[temp_frame].append(ant)

            frame = frames[0]
            x, y = xy.loc[id_exp, ant, frame]
            label_text = ax.text(
                x, y, '%i, %i' % (ant, frame), ha='center', va='bottom',
                color=color, fontsize=8, alpha=0)
            circ = plt.Circle((x, y), label_radius, color=color, alpha=0, fill=False)
            label_graph = ax.add_artist(circ)

            graph_dict[ant] = label_graph
            label_dict[ant] = label_text

        marking_df.groupby(id_ant_name).apply(do4each_group)

        frame0 = df_color.dropna(how='all').index[0]
        df_color = df_color.loc[frame0:]
        df_marking_label = df_marking_label.loc[frame0:]

        list_frames = df_marking_label.index
        list_frames = list(range(min(list_frames), max(list_frames)))

        frame_img = self.fig2data(fig)
        out = cv2.VideoWriter(
            filename=movie_add, fourcc=fourcc, fps=50, frameSize=(1920, 1080), isColor=True)
        out.write(frame_img)
        for f in list_frames:
            print(f)
            frame_img = movie.get_next_frame(grayscale=False)
            frame_graph.set_data(frame_img)

            for id_ant in list_ants:
                if id_ant in frame_ant_dict[f]:

                    c = df_color.loc[f, id_ant]
                    if isinstance(c, tuple):
                        frame_marking = df_marking_label.loc[f, id_ant]
                        x1, y1 = xy.loc[id_exp, id_ant, f]

                        graph_dict[id_ant].set_center([x1, y1])
                        graph_dict[id_ant].set_color(c[:-1])
                        graph_dict[id_ant].set_alpha(1)

                        label_dict[id_ant].set_position([x1, y1-label_radius])
                        label_dict[id_ant].set_color(c[:-1])
                        label_dict[id_ant].set_text('%i, %i' % (id_ant, frame_marking))
                        label_dict[id_ant].set_alpha(1)

                else:

                    graph_dict[id_ant].set_alpha(0)
                    label_dict[id_ant].set_alpha(0)

            frame_img = self.fig2data(fig)
            out.write(frame_img)

        out.release()

    @staticmethod
    def fig2data(fig, w=1080, h=1920):
        fig.canvas.draw()

        buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (w, h, 3)
        buf = np.roll(buf, 3, axis=2)
        return buf

    def get_movie(self, id_exp):
        address = self.get_movie_address(id_exp)
        return Movies(address, id_exp)

    def get_movie_address(self, id_exp):
        session, trial = self.exp.get_ref_id_exp(id_exp)
        return '%s%s/%s_S%02d_T%02d.MP4' % (root_movie, self.exp.group, self.exp.group, session, trial)

    @staticmethod
    def __draw_square(img, x, y, lg, c):
        for x0 in range(x-lg, x+lg+1):
            img[x0, y-lg] = c
            img[x0, y+lg] = c
        for y0 in range(y-lg, y+lg+1):
            img[x-lg, y0] = c
            img[x+lg, y0] = c

    @staticmethod
    def __draw_circle(img, x0, y0, lg, c):
        w = 1
        for theta in np.arange(-np.pi, np.pi+0.1, 0.1):
            x = int(x0+np.cos(theta)*lg)
            y = int(y0+np.sin(theta)*lg)
            for dx in range(-w, w+1):
                for dy in range(-w, w+1):
                    img[x+dx, y+dy] = c

    def movie_fast(self, id_exp, name_movie=None):

        label_radius = 50
        cmap_name = 'jet'
        dt_max = 10
        fnt_time = ImageFont.truetype('/Library/Fonts/Arial.ttf', 30)
        fnt_id = ImageFont.truetype('/Library/Fonts/Arial.ttf', 15)

        if name_movie is None:
            name_movie = 'movie%s' % id_exp
        movie_add = '%s%s/%s/%s.avi' % (root, self.exp.group, self.category, name_movie)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        print(movie_add)

        movie = self.exp.get_movie(id_exp)

        marking_name = 'potential_marking_intervals'
        self.exp.load(marking_name)
        xy_name = 'xy'
        self.exp.load_as_2d('mm10_x', 'mm10_y', xy_name, 'x', 'y')

        marking_df = self.exp.get_df(marking_name).loc[id_exp, :, :]
        xy = self.exp.get_df(xy_name).loc[id_exp, :, :]
        xy.x, xy.y = self.exp.convert_xy_to_movie_system(id_exp, xy.x, xy.y)

        self.exp.load(['crop_limit_x', 'crop_limit_y'])
        crop_x = list(self.exp.crop_limit_x.df.loc[id_exp].astype(int))
        crop_y = list(self.exp.crop_limit_y.df.loc[id_exp].astype(int))
        xy.x -= crop_x[0]
        xy.y -= crop_y[0]

        list_ants = list(set(list(marking_df.index.get_level_values(id_ant_name))))
        list_ants.sort()
        list_frames = xy.index.get_level_values(id_frame_name)
        list_frames = list(range(min(list_frames), max(list_frames)))

        df_marking_label = pd.DataFrame(index=list_frames, columns=list_ants)
        df_marking_label[:] = np.nan

        df_color = pd.DataFrame(index=list_frames, columns=list_ants)
        df_color[:] = np.nan

        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        fig.subplots_adjust(0, 0, 1, 1)
        ax.axis('equal')
        ax.invert_yaxis()
        ax.set_axis_off()

        cmap = plt.get_cmap(cmap_name)

        frame_ant_dict = {frame: [] for frame in list_frames}

        def do4each_group(df: pd.DataFrame):
            ant = df.index.get_level_values(id_ant_name)[0]
            frames = list(df.index.get_level_values(id_frame_name))
            dts = (df.values.ravel()*100).astype(int)

            color = cmap(np.random.randint(256))
            color = tuple((np.array(color)*255).astype(int))
            dt1 = min(dt_max, int(dts[0]/2))
            frame = frames[0]
            df_color.loc[frame - dt_max:frame + dt1, ant] = [color]
            df_marking_label.loc[frame - dt_max:frame + dt1, ant] = frame

            temp_frames = list(xy.loc[id_exp, ant, frame - dt_max:frame + dt1].index.get_level_values(id_frame_name))
            for frame in temp_frames:
                frame_ant_dict[frame].append(ant)

            for i, frame in enumerate(frames):
                color = cmap(np.random.randint(256))
                color = tuple((np.array(color)*255).astype(int))
                dt0 = min(dt_max, int(dts[i-1]/2))
                dt1 = min(dt_max, int(dts[i]/2))
                df_color.loc[frame - dt0:frame + dt1, ant] = [color]
                df_marking_label.loc[frame - dt0:frame + dt1, ant] = frame

                temp_frames = list(xy.loc[id_exp, ant, frame-dt_max:frame+dt1].index.get_level_values(id_frame_name))
                for temp_frame in temp_frames:
                    frame_ant_dict[temp_frame].append(ant)

        marking_df.groupby(id_ant_name).apply(do4each_group)

        frame0 = df_color.dropna(how='all').index[0]
        df_color = df_color.loc[frame0:]
        df_marking_label = df_marking_label.loc[frame0:]

        list_frames = df_marking_label.index
        list_frames = list(range(min(list_frames), max(list_frames)))

        frame_img = self.crop_img(movie.get_frame(frame0-1), crop_x, crop_y)
        frame_size = frame_img.shape[1], frame_img.shape[0]
        out = cv2.VideoWriter(
            filename=movie_add, fourcc=fourcc, fps=50, frameSize=frame_size, isColor=True)
        for f in list_frames:
            print(f)
            frame_img = self.crop_img(movie.get_next_frame(grayscale=False), crop_x, crop_y)
            frame_img = Image.fromarray(frame_img)

            draw_img = ImageDraw.Draw(frame_img)
            draw_img.text((0, 0), ' exp: %i\n frame: %i' % (id_exp, f), font=fnt_time, fill=(0, 0, 0))

            for id_ant in list_ants:
                if id_ant in frame_ant_dict[f]:

                    c = df_color.loc[f, id_ant]
                    if isinstance(c, tuple):
                        frame_marking = df_marking_label.loc[f, id_ant]
                        if np.isnan(frame_marking):
                            frame_marking = ''
                        else:
                            frame_marking = str(frame_marking)
                        x, y = xy.loc[id_exp, id_ant, f].astype(int)

                        text = '%i, %s' % (id_ant, frame_marking)
                        w, h = draw_img.textsize(text)
                        draw_img.text((x-w/2, int(y-label_radius-h*1.5)), text, font=fnt_id, fill=c)

                        draw_img.ellipse(
                            (x-label_radius, y-label_radius, x+label_radius, y+label_radius),
                            fill=None, outline=c, width=2)

            out.write(np.array(frame_img))

        out.release()

    @staticmethod
    def crop_img(img, x, y):
        return img[y[0]:y[1], x[0]:x[1]]
