import tkinter as tk
from tkinter.filedialog import askopenfilename


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd

from predictor import Predictor

TITLE = "Wikipedia traffic prediction"

class App(tk.Frame):
    def __init__(self, master):
        self.predictor = Predictor()

        tk.Frame.__init__(self, master)
        self.pack()
        self.master.title(TITLE)
        self.master.resizable(True, True)
        self.master.tk_setPalette(background='#ececec')

        self.master.protocol('WM_DELETE_WINDOW', self.click_cancel)
        self.master.bind('<Return>', self.click_ok)
        self.master.bind('<Escape>', self.click_cancel)

        x = (self.master.winfo_screenwidth() - self.master.winfo_reqwidth()) / 2
        y = (self.master.winfo_screenheight() - self.master.winfo_reqheight()) / 3
        # self.master.geometry("+{}+{}".format(x, y))
        self.master.geometry("800x600+30+30")

        self.master.config(menu=tk.Menu(self.master))

        # dialog_frame = tk.Frame(self)
        # dialog_frame.pack(padx=20, pady=15)

        # tk.Label(dialog_frame, text="This is your first GUI. (highfive)").pack()

        # show default image
        img_frame = tk.Frame(self, relief=tk.RAISED, borderwidth=1)
        img_frame.pack(pady=5)#fill=tk.BOTH, expand=True)
        # self.label = tk.Label(img_frame)
        # self.label.pack()#side="left")
        # self.showImage()
        self.figure = plt.Figure(figsize=(6,3), dpi=100)
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, img_frame)
        self.canvas.get_tk_widget().pack()

        self.mse = tk.StringVar()
        self.mse.set("Mean Absolute Error is: {}".format(0))
        t1 = tk.Label(self, textvariable=self.mse).pack()

        button_frame = tk.Frame(self)
        button_frame.pack()#padx=15, pady=(0, 15), anchor='e')

        # tk.Button(button_frame, text='OK', default='active', command=self.click_ok).pack(side='right')

        # tk.Button(button_frame, text='Close', command=self.click_cancel).pack(side='right')
        self.input_filename = tk.StringVar()

        # do a default file opening and prediction
        self.df = pd.read_csv('demo.csv').fillna(0)
        page_options = self.df['Page']
        self.predict()

        tk.Button(button_frame, text ='Open', command = self.inputCSV).pack(side="left")
        tk.Button(button_frame, text ='Predict', command = self.predict).pack(side="left")

        # radio button selections
        sel_major_frame = tk.Frame(self)
        sel_major_frame.pack()

        tk.Label(sel_major_frame, text="Select an interesting wiki page", font=("Helvetica", 16)).pack()

        sel_frame = tk.Frame(sel_major_frame)
        sel_frame.pack(padx=15, side="left")#, pady=(100, 15), anchor='e')
        self.selection = tk.IntVar()
        for i, button_item in enumerate(page_options):
            tk.Radiobutton(sel_frame,
                  text=button_item,
                  padx = 20,
                  variable=self.selection,
                  command=self.sel,
                  value=i).pack(anchor=tk.W)

        ## pick x day to predict in the future
        # dialog_frame = tk.Frame(sel_major_frame)
        # dialog_frame.pack(side="right")
        # dialog_frame.pack(padx=20, pady=15)

        # self.future_day = tk.StringVar()
        # self.future_day.set("Predict {} day traffic in future".format(5))
        # w2 = tk.Scale(dialog_frame, from_=1, to=14, command=self.update_future_day)#, orient=tk.HORIZONTAL)
        # w2.set(5)
        # w2.pack()
        # t1 = tk.Label(dialog_frame, textvariable=self.future_day).pack()

    def click_ok(self, event=None):
        print("The user clicked 'OK'")

    def click_cancel(self, event=None):
        print("The user clicked 'Cancel'")
        self.master.destroy()

    def update_future_day(self, val):
        self.future_day.set("Predict {} day traffic in future".format(val))

    def sel(self):
        selection = self.selection.get()
        self.predict(selection)

    def inputCSV(self):
        File = askopenfilename(title='Opne Input traffic trace')
        self.input_filename.set(File)
        self.df = pd.read_csv(self.input_filename.get()).fillna(0)
        self.predict()

    def predict(self, i = 0):
        target = self.df.iloc[i:i+1]
        self.ax.clear()
        mse = self.predictor.predict(target, self.ax)
        self.mse.set("Mean Absolute Error is: {}".format(mse))
        self.canvas.draw()

if __name__ == '__main__':

    root = tk.Tk()
    app = App(root)
    app.mainloop()
