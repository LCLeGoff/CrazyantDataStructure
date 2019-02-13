import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Scripts.root import root


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        exp = ExperimentGroupBuilder(root).build('UO')
        self.movie = exp.get_movie(1)
        fig, ax = plt.subplots(figsize=(1920/200., 1080/200.))
        fig.subplots_adjust(0, 0, 1, 1)
        ax.axis('off')
        ax.axis('equal')
        self.fig = fig
        self.ax = ax
        self.bg = self.plot_background()

        self.create_widgets()

    def create_widgets(self):

        self.bg_refresh_button()

        self.quit_button()

    def bg_refresh_button(self):
        bt = tk.Button(self)
        bt["text"] = "Refresh"
        bt["command"] = self.refresh_bg
        bt.pack(side="left")

    def plot_background(self):
        bg = self.ax.imshow(self.movie.get_frame(2091), cmap='gray')
        canvas = FigureCanvasTkAgg(self.fig, self)
        canvas.get_tk_widget().pack()
        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        return bg

    def quit_button(self):
        bt = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy)
        bt.pack(side="bottom")

    def refresh_bg(self):
        frame = 5000
        frame_img = self.movie.get_frame(frame)
        self.bg.set_array = frame_img
        canvas = FigureCanvasTkAgg(self.fig, self)
        canvas.get_tk_widget().pack()


app = Application(master=tk.Tk())
app.mainloop()
