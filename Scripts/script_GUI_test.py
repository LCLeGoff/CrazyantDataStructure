import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from DataStructure.Builders.ExperimentGroupBuilder import ExperimentGroupBuilder
from Scripts.root import root


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):

        exp = ExperimentGroupBuilder(root).build('UO')
        mov = exp.get_movie(1)
        mov.get_frame(2091)

        # tk.Frame.__init__(self, parent)
        # label = tk.Label(self, text="Graph Page!", font=LARGE_FONT)
        # label.pack(pady=10,padx=10)

        f, ax = plt.subplots()
        ax.imshow(mov.get_frame(2091), cmap='gray')

        canvas = FigureCanvasTkAgg(f, self)
        # canvas.draw()
        canvas.get_tk_widget().pack()

        # toolbar = NavigationToolbar2TkAgg(canvas, self)
        # toolbar.update()
        # canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Hello World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="left")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def say_hi(self):
        print("hi there, everyone!")


app = Application(master=tk.Tk())
app.mainloop()
