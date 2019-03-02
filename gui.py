import tkinter as tk

IMAGES = ['', 'iot', 'north_korea', 'clover', 'awaken']

class App(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.pack()
        self.master.title("Hello World")
        self.master.resizable(False, False)
        self.master.tk_setPalette(background='#ececec')

        self.master.protocol('WM_DELETE_WINDOW', self.click_cancel)
        self.master.bind('<Return>', self.click_ok)
        self.master.bind('<Escape>', self.click_cancel)

        x = (self.master.winfo_screenwidth() - self.master.winfo_reqwidth()) / 2
        y = (self.master.winfo_screenheight() - self.master.winfo_reqheight()) / 3
        # self.master.geometry("+{}+{}".format(x, y))
        self.master.geometry("800x600+30+30")

        self.master.config(menu=tk.Menu(self.master))

        dialog_frame = tk.Frame(self)
        dialog_frame.pack(padx=20, pady=15)

        tk.Label(dialog_frame, text="This is your first GUI. (highfive)").pack()

        # show default image
        self.label = tk.Label()
        self.showImage()

        # selections
        self.var = tk.IntVar()
        tk.Radiobutton(root,
              text="Internet-of-Thing",
              padx = 20,
              variable=self.var,
              command=self.sel,
              value=1).pack(anchor=tk.W)

        tk.Radiobutton(root,
              text="North_Korea_Nuclear",
              padx = 20,
              variable=self.var,
              command=self.sel,
              value=2).pack(anchor=tk.W)


        button_frame = tk.Frame(self)
        button_frame.pack(padx=15, pady=(0, 15), anchor='e')

        tk.Button(button_frame, text='OK', default='active', command=self.click_ok).pack(side='right')

        tk.Button(button_frame, text='Cancel', command=self.click_cancel).pack(side='right')

    def click_ok(self, event=None):
        print("The user clicked 'OK'")

    def click_cancel(self, event=None):
        print("The user clicked 'Cancel'")
        self.master.destroy()

    def sel(self):
        selection = self.var.get()
        self.showImage(selection)

    def showImage(self, idx = 1):
        img = tk.PhotoImage(file=IMAGES[idx]+".png")
        self.label.image = img
        self.label.configure(image=img)
        self.label.pack()

if __name__ == '__main__':

    root = tk.Tk()
    app = App(root)
    app.mainloop()
