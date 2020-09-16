# -*- coding: utf-8 -*-

"""
Created on Tue Jun 19 20:51:54 2018

@author: KangerJS

library for (linear) least square fitting


functions:

    fplsqGUI_2order()   function that creates a GUI for fitting the response function
                        of a damped second order harmonic oscillator
       
    fplsqGUI()      function that creates a GUI for fitting the function y=Ax+B
    
    fplsqAB()       function wrapper for curve_fit() scipy function for fitting y=Ax+B 
                    (no plot is generated)
    
    fplsqA()        function wrapper for curve_fit() scipy function for fitting y=Ax+B  
                    with B fixed (no plot is generated)
    
    fplsqB()        function wrapper for curve_fit() scipy function for fitting y=Ax+B, 
                    with A fixed (no plot is generated)

version 4.1.1       november 2019
- bug fix. Selecting weight option "standard deviation" was not properly handled

version 4.1         october 2019
- Added the function fplsqGUI_2order() to fit response of a damped harmonic oscillator

version 4.0         august 2019
- GUI migrated to PyQt5
- Included fixed intercept fitting with arbitrary value for intercept
- Use curve_fit from the scipy.optimize package
- new range selector
- more general implementation to fit arbitrary functions for future updates
- removed chisquare from statistical properties
- included t-value (95% conf.interv)
- included minimum sum of (weighted) residuals

version 3.0         oktober 2018
update: 
- not using the scipy optimising package for fitting, but direct calculations using inverse method.
- included a data range selector to select the range of data used for fitting
- layout of textboxes and checkboxes improved
- changed radiobuttons to checkbox for weighted fit
- included chi-square and reduced chi-square calculation
- included degrees-of-freedom in display of fitparameters

Version 2.0         july 2018
- First working version of fplsqlib 
- based on the matlab version fmlsq by J. v.d. Meulen

UTwente Applied Physics
j.s.kanger june 2018
    
"""

__all__ = ['fplsqGUI', 'fplsqAB',  'fplsqA', 'fplsqB']

# import the required packages
import warnings
import inspect
import sys
from functools import wraps

# numpy and scipy packages
import numpy as np
from numpy import sum
from scipy import stats
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

# PyQt5 pacakages
from PyQt5 import QtCore, QtWidgets, QtGui, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Matplotlib packages
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import  SpanSelector
from matplotlib.path import Path
import matplotlib.patches as patches

WEIGHTOPTIONS = ('none', 'relative', 'standard deviation')
VERSION_INFO = 'FPLSQ versie 4.1.1'
MODEL_NUMPOINTS = 1000


class Model:
    """ class to store and define the model used for fitting """
    def __init__(self, func, name, jac=None):
        self.func = func
        self.name = name
        self.jac = jac
        __args = inspect.signature(func).parameters
        args = [arg.name for arg in __args.values()]
        self.pars = [dict(name=arg, value=1., stderr=0., fixed=False) for arg in args[1:]]

    def pars_to_dict(self):
        parsdict = {par['name'] : {key:val for key, val in par.items() if key != 'name'} for par in self.pars}
        return parsdict

    @classmethod
    def linear(cls):
        return cls(lambda x, A, B: A*x+ B, 'y = Ax + B')

    @classmethod
    def quadratic(cls):
        return cls(lambda x, A, B, C: A*x**2+ B*x + C, 'y = Ax^2 + Bx + C')

    @classmethod
    def exponential(cls):
        return cls(lambda x, Amp, Tau, Offset: Amp*np.exp(x/Tau)+Offset, 'Amp*exp(x/Tau)+Offset')

class Data:
    """ class to store the data used for fitting """
    def __init__(self, x, y, yerr=None):
        self.set(x, y, yerr)
    
    def set(self, x, y, yerr=None):
        self.x = x 
        self.y = y 
        self.yerr = yerr 

    def get(self, xrange=[-np.inf, np.inf]):
        """ returns x, y and error data for which the x-value lies in the range provided by range  """
        (xmin, xmax) = xrange
        datamask = [(xmin <= x <= xmax) for x in self.x]
        yerr = self.yerr[datamask] if self.yerr is not None else None
        return self.x[datamask], self.y[datamask], yerr


class DraggableVLine:
    """ class to create a draggable vertical line in a plot """
    def __init__(self, line, xdata):
        self.line = line
        self.press = None
        self.xdata = xdata

    def find_nearest(self, x):
            idx = (np.abs(self.xdata - x)).argmin()
            return self.xdata[idx]

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.line.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.line.axes: return
        contains, _ = self.line.contains(event)
        if not contains: return
        x,_ = self.line.get_xdata()
        self.press = x, event.xdata

    def on_motion(self, event):
        if self.press is None: return
        if event.inaxes != self.line.axes: return
        x, xpress = self.press
        dx = event.xdata - xpress
        x_clip = self.find_nearest(x+dx)
        self.line.set_xdata([x_clip, x_clip])
        self.line.figure.canvas.draw()

    def on_release(self, event):
        self.press = None
        self.line.figure.canvas.draw()

    def disconnect(self):
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)


class RangeSelector:
    """ class that creates a rangeselector in a plot consisting of two draggable vertical lines """
    def __init__(self, ax, xdata):
        self.ax = ax
        self.xdata = xdata
        self.xmin = np.min(xdata)
        self.xmax = np.max(xdata)
        self.vline1 = self.ax.axvline(x=self.xmin, linewidth=4, linestyle='--', color='gray')
        self.vline2 = self.ax.axvline(x=self.xmax, linewidth=4, linestyle='--', color='gray')
        self.dvl = DraggableVLine(self.vline1, xdata)
        self.dvl.connect()
        self.dv2 = DraggableVLine(self.vline2, xdata)
        self.dv2.connect()

    def get_range(self):
        x1,_ = self.vline1.get_xdata()
        x2,_ = self.vline2.get_xdata()
        if x1>x2:
            x1, x2 = x2, x1
        return x1, x2

    def __remove__(self):
        self.vline1.remove()
        self.vline2.remove()


class PlotWidget(QtWidgets.QWidget):
    """ Qt widget to hold the matplotlib canvas and the tools for interacting with the plots """
    def __init__(self, data, xlabel, ylabel):
        QtWidgets.QWidget.__init__(self)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.canvas = PlotCanvas(data, xlabel, ylabel)        
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.addSeparator()
        
        self.ACshowselector = QtWidgets.QAction('Activate/Clear RangeSelector')
        self.ACshowselector.setIconText('RANGE SELECTOR')
        self.ACshowselector.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        self.ACshowselector.triggered.connect(self.toggle_showselector)

        self.toolbar.addAction(self.ACshowselector)
        self.toolbar.addSeparator()
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)

    def toggle_showselector(self):
        self.canvas.toggle_rangeselector()

class PlotCanvas(FigureCanvas):
    """ class to hold a canvas with a matplotlib figure and two subplots for plotting data and residuals """
    def __init__(self, data, xlabel, ylabel):
        self.data = data  # contains the x, y and error data
        self.fitline = None  # contains the fitline if available
        self.residuals = None  # contains the residuals if available

        # setup the FigureCanvas
        self.fig = Figure()
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)  

        # init some statevars
        self.errorbox_patch = None  # to hold the patches for drawing errorboxes
        self.range_selector = None

        # create the figure and axes       
        gs = self.fig.add_gridspec(3, 1)  # define three rows and one column
        self.ax1 = self.fig.add_subplot(gs[0:2,0])  # ax1 holds the plot of the data and spans two rows
        self.ax2 = self.fig.add_subplot(gs[2,0], sharex=self.ax1)  # ax2 holds the plot of the residuals and spans one row
        self.ax1.grid()
        self.ax1.set_ylabel(ylabel)
        self.ax2.axhline(y=0, linestyle='--', color='black')    
        self.ax2.set_ylabel('residual')
        self.ax2.set_xlabel(xlabel)
        self.ax2.grid() 

        # create empty lines for the data, fit and residuals
        self.data_line, = self.ax1.plot([], [], color='black', marker='o', fillstyle='none', lw=0, label='data')
        self.fitted_line, = self.ax1.plot([], [], label='fitted curve', linestyle='--', color='black') 
        self.residual_line, = self.ax2.plot([],[], color='k', marker='.', lw=1)    
        self.ax1.legend()  

    def toggle_rangeselector(self):
        if self.range_selector is None:
            self.range_selector = RangeSelector(self.ax1, self.data.x)
            self.redraw()
        else:
            self.range_selector.__remove__()
            self.range_selector = None
            self.redraw()
    
    def get_range(self):
        if self.range_selector is None: return (-np.inf, np.inf)
        return self.range_selector.get_range()

    def update_plot(self):        
        # update the plotlines
        self.data_line.set_data(self.data.x, self.data.y) 
        if self.residuals is not None:
            self.residual_line.set_data(self.data.x, self.residuals)
        else:
            self.residual_line.set_data(self.data.x[0], 0)  # need one datapoint for autoscaling to work
        if self.fitline is not None: self.fitted_line.set_data(self.fitline[0], self.fitline[1])
        
        # update errorboxes
        if self.data.yerr is not None:
            # remove any previous errorbar patch
            if self.errorbox_patch:
                self.ax1.patches.remove(self.errorbox_patch)

            # calculate new errorbar info, patch it in
            verts = []
            codes = []      
            for (x, y, e) in zip(self.data.x, self.data.y, self.data.yerr):
                verts.append((x, y+e))
                codes.append(Path.MOVETO)
                verts.append((x, y-e))
                codes.append(Path.LINETO)
            barpath = Path(verts, codes)
            self.errorbox_patch = patches.PathPatch(barpath, lw=1, edgecolor='k', zorder=2)
            self.ax1.add_patch(self.errorbox_patch)

        # rescale the axis
        self.ax1.relim()
        self.ax1.autoscale()
        self.ax2.relim()
        self.ax2.autoscale()

        # make the min and max yscale limits of the residual plot equal
        ymax = max(np.abs(self.ax2.get_ylim()))
        self.ax2.set_ylim(-ymax, ymax)

        # draw the plot
        self.fig.tight_layout()
        self.redraw()    

    def redraw(self):
        self.fig.canvas.draw() 


class ParView(QtWidgets.QWidget):
    """ Qt widget to show and change a fitparameter """
    def __init__(self, par):
        QtWidgets.QWidget.__init__(self)  
        self.par = par
        self.label = QtWidgets.QLabel(par['name'])
        self.edit = QtWidgets.QLineEdit('')
        self.update_value()
        self.check = QtWidgets.QCheckBox('fix')
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.edit)
        layout.addWidget(self.check)
        self.setLayout(layout)

    def read_value(self):
        """ read userinput (value and fixed) in the parameter data """
        self.par['value'] = float(self.edit.text())
        self.par['fixed'] = self.check.isChecked()
        return None

    def update_value(self):
        value = self.par['value']
        self.edit.setText(f'{value:1.5E}')
        return None


class ModelView(QtWidgets.QGroupBox):
    """ Qt widget to show and control the fit model """
    def __init__(self, model):
        self.model = model
        QtWidgets.QGroupBox.__init__(self, 'Model settings')
        VBox = QtWidgets.QVBoxLayout()
        HBox = QtWidgets.QHBoxLayout()
        
        #self.model_info = QtWidgets.QTextEdit(self.model.name)  
        #self.model_info.setReadOnly(True)

        self.parviews = [ParView(par) for par in self.model.pars]
        self.WeightLabel = QtWidgets.QLabel('Weighted Fit:')
        self.Yweightcombobox = QtWidgets.QComboBox()
        self.Yweightcombobox.addItems(WEIGHTOPTIONS)
        HBox.addWidget(self.WeightLabel)
        HBox.addWidget(self.Yweightcombobox)
        HBox.addStretch(1)
        
        #VBox.addWidget(self.model_info)
        for parview in self.parviews:
            VBox.addWidget(parview)
        VBox.addLayout(HBox)
        
        self.setLayout(VBox)

    def disable_weight(self):
        self.Yweightcombobox.setDisabled(True)

    def enable_weight(self):
        self.Yweightcombobox.setEnabled(True)

    def get_weight(self):
        return self.Yweightcombobox.currentText()

    def set_weight(self, weight):
        index = self.Yweightcombobox.findText(weight, QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.Yweightcombobox.setCurrentIndex(index)

    def read_values(self):
        """ reads values from userinput into the model """
        for parview in self.parviews:
            parview.read_value()
        return None
        
    def update_values(self):
        for parview in self.parviews:
            parview.update_value()
        return None


class ReportView(QtWidgets.QTextEdit):
    """ prints a fitreport in a non-editable textbox. Report should be a (nested) dictionary """
    def __init__(self):
        QtWidgets.QTextEdit.__init__(self, 'none')  
        self.setReadOnly(True)
  
    def update_report(self, fitreport):
        """ updates the text of the texteditbox with the content of a (nested) dictionary fitreport """
        def print_dict(adict, level):
            for key, item in adict.items():
                if type(item) is dict:
                    if level == 1: self.insertPlainText('======== ')
                    self.insertPlainText(str(key))
                    if level == 1: self.insertPlainText(' ======== ')
                    self.insertPlainText('\n\n')
                    print_dict(item, level + 1)
                else:
                    self.insertPlainText(str(key) + '\t\t: ' + str(item) + '\n')
            self.insertPlainText('\n')

        self.clear()
        print_dict(fitreport, 1)    
        

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, model, data, xlabel, ylabel):    
        super(MainWindow , self).__init__()
        self.model = model
        self.data = data
        self.xlabel, self.ylabel = xlabel, ylabel   
        self.initGUI()

        # perform some initial default settings
        if self.data.yerr is None:
            self.modelview.disable_weight()
        else:
            self.modelview.set_weight('relative')
        self.plotwidget.canvas.update_plot()
        

    def closeEvent(self, event):
        """needed to properly quit when running in IPython console / Spyder IDE"""
        QtWidgets.QApplication.quit()

    def initGUI(self):
        # main GUI proprieties
        self.setGeometry(100, 100, 1415, 900)
        self.setWindowTitle(VERSION_INFO)
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        # creating the required widgets
        self.plotwidget = PlotWidget(self.data, self.xlabel, self.ylabel)  # holds the plot
        self.modelview = ModelView(self.model)  # shows the model and allows users to set fitproperties
        self.fitbutton = QtWidgets.QPushButton('FIT', clicked = self.fit) 
        self.reportview = ReportView()  # shows the fitresults
               
        # create a frame with a vertical layout to organize the modelview, fitbutton and reportview
        self.fitcontrolframe = QtWidgets.QGroupBox()
        fitcontrollayout = QtWidgets.QVBoxLayout()
        for widget in (self.modelview, self.fitbutton, self.reportview):
            fitcontrollayout.addWidget(widget)
        self.fitcontrolframe.setLayout(fitcontrollayout)
        
        # putting it all together: Setup the main layout
        mainlayout = QtWidgets.QHBoxLayout(self._main)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.plotwidget)
        splitter.addWidget(self.fitcontrolframe)
        mainlayout.addWidget(splitter)
                 
      
    def showdialog(self, message, info='', details=''):
        """ shows an info dialog """
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(message)
        msg.setInformativeText(info)
        msg.setWindowTitle("Message")
        msg.setDetailedText(details)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    
    def fit(self):
        """ updates the model performs the fit and updates the widgets with the results """
        # update the modelvalues from userinput 
        try:
            self.modelview.read_values()
        except ValueError:
            self.showdialog('Not a valid input initial parameter values')
            return None

        # collect modelinfo
        func = self.model.func
        jac = self.model.jac
        p0 = [par['value'] for par in self.model.pars]
        pF = [par['fixed'] for par in self.model.pars]
        weighted = self.modelview.get_weight()   

        # extract data to match the range provided by the user
        x, y, ye = self.data.get(self.plotwidget.canvas.get_range())
        if weighted == 'none':
            ye = np.ones(len(y))  # error of 1 is equal to no weights
    
        # check if there are free fitpars
        numpar = sum([1 for f in pF if not f])
        if numpar == 0:
            self.showdialog("No free fit parameters.")
            return None
        
        # check if there are enough degrees of freedom
        dof = int(len(x) - numpar)
        if dof <= 0:
            self.showdialog("The number of degrees of freedom (dof) should be at least one." + \
                            " Try to increase the number of datapoints or to increase the number of free fitparameters.")
            return None

        # perform the fit
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)  # make sure the OptimizeWarning is raised as an exception
            try:
                fitpars, fitcov = curve_fit_wrapper(func, x, y, p0=p0, pF=pF, sigma=ye, absolute_sigma=weighted=='standard deviation', jac=jac)
            
            except (ValueError, RuntimeError, OptimizeWarning):
                self.showdialog(str(sys.exc_info()[1]))

            else:
                # update parameters of the model
                stderrors = np.sqrt(np.diag(fitcov))
                for par, fitpar, stderr in zip(self.model.pars, fitpars, stderrors):
                    par['value'] = fitpar
                    par['stderr'] = stderr

                # create a fitreport
                fitreport =    {
                                'FITPARAMETERS'         : {
                                                            'model'              : self.model.name,
                                                            'weight'             : weighted,
                                                            'N'                  : len(x),
                                                            'dof'                : dof,
                                                            't95-value'          : stats.t.ppf(0.975, dof)
                                                        },
                                'FITRESULTS'            : self.model.pars_to_dict(), 
                                'STATISTICS'            : {
                                                            'Smin'               : sum( ((y - func(x, *fitpars))/ye)**2 )
                                                        }
                                }

                # update the widgets
                self.modelview.update_values()
                self.reportview.update_report(fitreport)
                xfit = np.linspace(self.data.x.min(), self.data.x.max(), MODEL_NUMPOINTS)
                self.plotwidget.canvas.fitline = [xfit, func(xfit,*fitpars)]
                self.plotwidget.canvas.residuals = self.data.y - func(self.data.x,*fitpars)
                self.plotwidget.canvas.update_plot() 


def curve_fit_wrapper(func, *pargs, p0=None, pF=None, jac=None, **kwargs):
    """ wrapper around the scipy curve_fit() function to allow parameters to be fixed """

    # extract arguments of the function func
    __args = inspect.signature(func).parameters
    args = [arg.name for arg in __args.values()]

    # populate pF and p0 to default if not provided in kwargs
    if pF is None: pF = np.array([False for _ in args[1:]])  # set all parameters to free
    if p0 is None: p0 = np.array([1 for _ in args[1:]])  # set all init values to 1

    # make lists of new function arguments and function arguments to be passed to original function
    newfunc_args = [args[0]] + [arg for arg, fix in zip(args[1:], pF) if not fix]
    orifunc_args = [args[0]] + [arg if not fix else str(p) for arg, fix, p in zip(args[1:], pF, p0)]

    # make a string defining the new function as a lambda expression and evaluate to function
    fit_func = eval(f"lambda {', '.join(newfunc_args)} : func({', '.join(orifunc_args)})", locals())

    # make a string defining the new jacobian function (if specified) as a lambda expression and evaluate to function
    if callable(jac):
        indices = np.array([index for index, value in enumerate(pF) if value==False])
        fit_jac = eval(f"lambda {', '.join(newfunc_args)} : jac({', '.join(orifunc_args)})[:, indices]", locals())
    else:
        fit_jac = jac

    # populate a list of initial values for free fit-parameters
    p0_fit = np.array([p for p, fix in zip(p0, pF) if not fix])
    
    # peform the fit with the reduced function
    popt, cov = curve_fit(fit_func, *pargs, p0=p0_fit, jac=fit_jac, **kwargs)
    
    # rebuild the popt and cov to include fixed parameters
    p0_fix = [p for p, fix in zip(p0,pF) if fix]  # values of fixed parameters
    id_fix = np.where(pF)[0]  # indices of fixed parameters
    for id, p in zip(id_fix, p0_fix):
        popt = np.insert(popt, id, p, axis=0)  # fill in the popt at the free fit parameters

    # rebuild covariance matrix to include both fixed and optimized pars
    for id in id_fix:
        cov = np.insert(cov, id, 0, axis=1)  # add zero rows and columns for fixed par
        cov = np.insert(cov, id, 0, axis=0)

    return popt, cov    

def validate_input(func):
    @wraps(func)  # to get the docstring of the decorated function and proper working of help and ?
    def func_wrapper(xdata, ydata, *args, **kwargs):
        for var in [xdata, ydata]:
            if type(var) is not np.ndarray:
                raise Exception('data should have type numpy array')
        if len(xdata) != len(ydata):
            raise Exception('xdata and ydata should be of equal length')
        if 'yerr' in kwargs:
            if type(kwargs['yerr']) is not np.ndarray:
                raise Exception('data should have type numpy array')
            if len(kwargs['yerr']) != len(ydata):
                raise Exception('yerr and ydata should be of equal length')
        res = func(xdata, ydata, *args, **kwargs)
        return res
    return func_wrapper

@validate_input
def fplsqAB(xdata, ydata, yerr=None):
    """
    performs a linear least squares fit of the form y = Ax + B.
    
    Arguments:
    xdata  -- numpy array: with x-coordinates of the data
    ydata  -- numpy array: with y-coordinates of the data
        
    Keyword arguments:
    yerr -- numpy array, default:None   used for weighted fit with
                                        a relative weight defined 
                                        as 1/yerr**2   

    returns:
    tuple of size 2 containing two numpy arrays:
         np.array([A, B]) and np.array([sA, sB]),
         with sA and sB the standard errors in A en B
          
    example of use:
        
        #define x and y data as 1 dimensional numpy arrays of equal length
        xdata = np.array([1,2,3,4,5])
        ydata = np.array([-3.5, -2.4, -1, 0.5, 1.8])
        
        #optinally define the errors in the ydata
        yerr = np.array([0.5,0.4,0.6,0.5,0.8])
        
        #execute the function
        fitpar, fitsd = fplsqAB(xdata, ydata, yerr=yerr)
        
        #fitpar is an numpy array with the values for A and B
        #fitsd is an numpy array with the standard deviations in A and B  
    """
    # estimate intial parameters
    slope = (ydata.max()-ydata.min())/(xdata.max()-xdata.min())
    offset = ydata.mean() - slope*xdata.mean()

    func = lambda x, a, b: a*x + b
    par, cov = curve_fit_wrapper(func, xdata, ydata, p0=[slope, offset], pF=[False, False], sigma=yerr, absolute_sigma=False)
    return (par[0], par[1]), (np.sqrt(cov[0,0]), np.sqrt(cov[1,1]))

@validate_input
def fplsqA(xdata, ydata, yerr=None, offset=0):
    """
    performs a linear least squares fit of the form y = Ax + offset.
    
    Arguments:
    xdata  -- numpy array: with x-coordinates of the data
    ydata  -- numpy array: with y-coordinates of the data
        
    Keyword arguments:
    offset -- float: default 0
    yerr -- numpy array, default:None   used for weighted fit with
                                        a relative weight defined 
                                        as 1/yerr**2    
    returns:
    tuple of size 2 containing two numpy arrays:
         np.array([A, B]) and np.array([sA, sB]),
         with sA and sB the standard errors in A en B
     
    
    example of use:
        
        #define x and y data as 1 dimensional numpy arrays of equal length
        xdata = np.array([1,2,3,4,5])
        ydata = np.array([-3.5, -2.4, -1, 0.5, 1.8])
        
        #optinally define the errors in the ydata
        yerr = np.array([0.5,0.4,0.6,0.5,0.8])
        
        #execute the function
        fitpar, fitsd = fplsqA(xdata, ydata, yerr=yerr)
        
        #fitpar is an numpy array with the value for A 
        #fitsd is an numpy array with the standard error in A    
    """
    # estimate intial parameters
    slope = (ydata.mean() - offset)/xdata.mean()

    func = lambda x, a, b: a*x + b
    par, cov = curve_fit_wrapper(func, xdata, ydata, p0=[slope, offset], pF=[False, True], sigma=yerr, absolute_sigma=False)
    return par[0], np.sqrt(cov[0,0])

@validate_input
def fplsqB(xdata, ydata, slope, yerr=None):
    """
    performs a linear least squares fit of the form y = Ax + B, with A fixed
    
    Arguments:
    xdata  -- numpy array: with x-coordinates of the data
    ydata  -- numpy array: with y-coordinates of the data
    slope   -- float:       fixed value for the slope (A-value in Ax+B)
    
    Keyword arguments:
    yerr -- numpy array, default:None   used for weighted fit with
                                        a relative weight defined 
                                        as 1/yerr**2   
    
    returns:
    tuple of size 2 containing two numpy arrays:
         np.array([A, B]) and np.array([sA, sB]),
         with sA and sB the standard errors in A en B
         
    
    example of use:
        
        #define x and y data as 1 dimensional numpy arrays of equal length
        xdata = np.array([1,2,3,4,5])
        ydata = np.array([-3.5, -2.4, -1, 0.5, 1.8])
        
        #optinally define the errors in the ydata
        yerr = np.array([0.5,0.4,0.6,0.5,0.8])
        
        #define the slope (value for A)
        fixed_slope = 2
        
        #execute the function
        fitpar, fitsd = fplsqB(xdata, ydata, fixed_slope, yerr=y_error)
        
        #fitpar is an numpy array with the value for B
        #fitsd is an numpy array with the standard error in B
    """ 
    # estimate intial parameters
    offset = ydata.mean() - slope*xdata.mean()

    func = lambda x, a, b: a*x + b
    par, cov = curve_fit_wrapper(func, xdata, ydata, p0=[slope, offset], pF=[True, False], sigma=yerr, absolute_sigma=False)
    return par[1], np.sqrt(cov[1,1])   



@validate_input
def fplsqGUI(xdata, ydata, yerr=None, xlabel='x-values', ylabel='y_values'):   
    """
    Graphical user interface for linear fitting.
    
    Arguments:
    xdata  -- numpy array: with x-coordinates of the data
    ydata  -- numpy array: with y-coordinates of the data
    
    Keyword arguments:
    yerr -- numpy array, default:None   used for weighted fit with
                                        a relative weight defined 
                                        as 1/yerr**2  
    xlabel  -- string, default:'x-values'   x-axis title
    ylabel  -- string, default:'y-values'   y-axis title

    Returns:
    None
                                         
    example of use:
        
        #define x and y data as 1 dimensional numpy arrays of equal length
        xdata = np.array([1,2,3,4,5])
        ydata = np.array([-3.5, -2.4, -1, 0.5, 1.8])
        
        #optinally define the errors in the ydata
        yerr = np.array([0.5,0.4,0.6,0.5,0.8])
        
        #optionally define axis titles
        xlabel = 'time / s'
        ylabel = 'height / m'
        
        #execute the function
        fplsqGUI(xdata, ydata, yerr=yerr, xlabel=xlabel, ylabel=ylabel)
    
    """  

    # create model and data object
    def func(x, a, b):
        """
        y = ax + b
        """
        return a*x + b
    _curve_fit_GUI(func, xdata, ydata, yerr=yerr, xlabel=xlabel, ylabel=ylabel)  
    return None

@validate_input
def fplsqGUI_2order(xdata, ydata, yerr=None, xlabel='x-values', ylabel='y_values'):   
    """
    Graphical user interface for fitting the response of a damped harmonic oscillator 
    y = 1/(sqrt((1-(w/w0)**2)**2+(2*b*w/w0)**2)).
    w  : frequency
    b  : damping
    w0 : resonanance frequency

    
    Arguments:
    xdata  -- numpy array: with x-coordinates of the data
    ydata  -- numpy array: with y-coordinates of the data
    
    Keyword arguments:
    yerr -- numpy array, default:None   used for weighted fit with
                                        a relative weight defined 
                                        as 1/yerr**2  
    xlabel  -- string, default:'x-values'   x-axis title
    ylabel  -- string, default:'y-values'   y-axis title

    Returns:
    None
                                         
    example of use:
        
        #define x and y data as 1 dimensional numpy arrays of equal length
        xdata = np.array([1,2,3,4,5])
        ydata = np.array([-3.5, -2.4, -1, 0.5, 1.8])
        
        #optinally define the errors in the ydata
        yerr = np.array([0.5,0.4,0.6,0.5,0.8])
        
        #optionally define axis titles
        xlabel = 'time / s'
        ylabel = 'height / m'
        
        #execute the function
        fplsqGUI(xdata, ydata, yerr=yerr, xlabel=xlabel, ylabel=ylabel)
    
    """  

    # create model and data object
    # define fitfunction
    def func(w,b,w0):
        """
        2de - orde systeem
        b  : demping
        w0 : resonantie frequentie
        1/(np.sqrt((1-(w/w0)**2)**2+(2*b*w/w0)**2))
        """
        return 1/(np.sqrt((1-(w/w0)**2)**2+(2*b*w/w0)**2))


    _curve_fit_GUI(func, xdata, ydata, yerr=yerr, xlabel=xlabel, ylabel=ylabel)  
    return None


def _curve_fit_GUI(func, xdata, ydata, yerr=None, xlabel='x-values', ylabel='y_values'):   
    """
    helper function that executes the GUI with the proper data and model 
    """

    # create model and data objects
    if callable(func):
        model_name = func.__doc__
        mymodel = Model(func, model_name)
    else:
        raise Exception('Not a valid function')

    mydata = Data(xdata, ydata, yerr)

    # execute the GUI
    app = QtWidgets.QApplication([])
    MyApplication = MainWindow(mymodel, mydata, xlabel, ylabel)
    MyApplication.show()
    app.exec_()  
    return None




if __name__ == "__main__":
    # example of use and testing"
    print('Running GUI with some test data')

    # create test data
    U0 = 2.12
    U = np.array([2.12,2.24,2.4,2.8,3.48,4.88,5.68,4.16,2.68,1.8,1.32,1.08,0.88,0.544,0.456,0.368,0.132])/U0 
    f = np.array([465,925,1420,1905,2398,2894,3458,3956,4439,4990,5560,6024,6527,7541,8130,9141,14270])   

    # execute the GUI version of fplsq
    fplsqGUI_2order(f, U, xlabel='Frequency [rad/s]', ylabel='Normalized Voltage U/U0')

