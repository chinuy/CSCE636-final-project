import tkinter as tk
from tkinter.filedialog import askopenfilename

IMAGES = ['iot', 'north_korea', 'clover', 'awaken']
mock_botton = ["Internet-of-Thing", "North_Korea_Nuclear", "Clover-10", "Movie-Awaken my Love"]
TITLE = "Wikipedia traffic prediction"

class App(tk.Frame):
    def __init__(self, master):
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
        # tk.Label(dialog_frame, text="Wikipedia traffic prediction").pack()

        # show default image
        img_frame = tk.Frame(self, relief=tk.RAISED, borderwidth=1)
        img_frame.pack()#fill=tk.BOTH, expand=True)
        self.label = tk.Label(img_frame)
        self.label.pack()#side="left")
        self.showImage()

        # radio button selections
        sel_major_frame = tk.Frame(self)
        sel_major_frame.pack()

        tk.Label(sel_major_frame, text="Select an interesting wiki page", font=("Helvetica", 16)).pack()

        sel_frame = tk.Frame(sel_major_frame)
        sel_frame.pack(padx=15, side="left")#, pady=(100, 15), anchor='e')
        self.var = tk.IntVar()
        for i, button_item in enumerate(mock_botton):
            tk.Radiobutton(sel_frame,
                  text=button_item,
                  padx = 20,
                  variable=self.var,
                  command=self.sel,
                  value=i).pack(anchor=tk.W)

        ## pick x day to predict in the future

        dialog_frame = tk.Frame(sel_major_frame)
        dialog_frame.pack(side="right")
        # dialog_frame.pack(padx=20, pady=15)

        self.future_day = tk.StringVar()
        self.future_day.set("Predict {} day traffic in future".format(5))
        w2 = tk.Scale(dialog_frame, from_=1, to=14, command=self.update_future_day)#, orient=tk.HORIZONTAL)
        w2.set(5)
        w2.pack()
        t1 = tk.Label(dialog_frame, textvariable=self.future_day).pack()


        button_frame = tk.Frame(self)
        button_frame.pack()#padx=15, pady=(0, 15), anchor='e')

        # tk.Button(button_frame, text='OK', default='active', command=self.click_ok).pack(side='right')

        # tk.Button(button_frame, text='Close', command=self.click_cancel).pack(side='right')
        self.input_filename = tk.StringVar()
        tk.Button(button_frame, text ='Open', command = self.inputCSV).pack(side="left")
        tk.Button(button_frame, text ='Predict', command = self.Predict).pack(side="left")

    def click_ok(self, event=None):
        print("The user clicked 'OK'")

    def click_cancel(self, event=None):
        print("The user clicked 'Cancel'")
        self.master.destroy()

    def update_future_day(self, val):
        self.future_day.set("Predict {} day traffic in future".format(val))

    def sel(self):
        selection = self.var.get()
        self.showImage(selection)

    def showImage(self, idx = 0):
        img = tk.PhotoImage(file=IMAGES[idx]+".png")
        self.label.image = img
        self.label.configure(image=img)

    def inputCSV(self):
        File = askopenfilename(title='Opne Input traffic trace')
        self.input_filename.set(File)

    def Predict(self):
        print("user hit the prediction button")
        # model = load_model(".h5")
        # model.predict()

if __name__ == '__main__':

    root = tk.Tk()
    app = App(root)
    app.mainloop()
