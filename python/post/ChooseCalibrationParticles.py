import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

import tkinter as tk
from tkinter import ttk

from scipy.interpolate import interp1d

import ReadCPIFiles
import ReadPositionFile

"""
    Read the CPI files
"""
(times,foc,len1,im1,boundaries,x,y)=ReadCPIFiles.ReadMAT( \
    pathName='/Users/mccikpc2/Downloads/CPI_cals/CPICalibration/Cal210302')
"""
    Read the positional calibration data
"""
(posx,posy,timesp)=ReadPositionFile.ReadFile(\
    fileName='/Users/mccikpc2/Downloads/CPI_cals/Matlab/ALL_DAYS_POSITION_TIME_4.txt')


PosXInt = interp1d(timesp,posx,kind='previous',fill_value='extrapolate')
PosYInt = interp1d(timesp,posy,kind='previous',fill_value='extrapolate')   

posxx=PosXInt(times) 
posyy=PosYInt(times) 
        











LARGE_FONT= ("Verdana", 12)

def line_select_callback(eclick, erelease):
    """
        eclick and erelease are the press and release events
    """
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    print(" The button you used were: %s %s" % (eclick.button, erelease.button))

def toggle_selector(event):
    print(' Key pressed.')
    if event.char in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.char in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)



class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        #tk.Tk.iconbitmap(self, default="@clienticon.xbm")
        tk.Tk.wm_title(self, "CPI calibration helper")
        
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo, PageThree):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="Visit Page 1",
                            command=lambda: controller.show_frame(PageOne))
        button.pack()

        button2 = ttk.Button(self, text="Visit Page 2",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()

        button3 = ttk.Button(self, text="Graph Page",
                            command=lambda: controller.show_frame(PageThree))
        button3.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page One!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page Two",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page Two!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page One",
                            command=lambda: controller.show_frame(PageOne))
        button2.pack()


def update_plot(pos):
    import numpy as np
    a1=[1,2,3,4,5,6,7,8]
    a2=np.random.rand(8)
    a.plot(a1,a2)
    canvas.draw()


class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        global a
        global canvas
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Graph Page!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        a.plot(posxx,foc,'.')
#         a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])

        

        canvas = FigureCanvasTkAgg(f, self)
        toggle_selector.RS = RectangleSelector(a, line_select_callback,
                                       drawtype='box',useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)

        #canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

#         canvas.mpl_connect('key_press_event', toggle_selector)
        

        w = tk.Scale(self, from_=0, to=200, orient=tk.HORIZONTAL, \
            command=update_plot) 
        w.pack() 
        

app = SeaofBTCapp()
app.bind( "<Key>",toggle_selector)
app.mainloop()