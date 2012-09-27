#!/usr/bin/env python

from traits.api import HasTraits, Int, Range, Array, List, Property, Enum
from enthought.traits.ui.api import View, Group, VGroup, HGroup, Item
from chaco.chaco_plot_editor import ChacoPlotItem
import numpy as np
import pyfluids


class RiemannApp(HasTraits):
    xdat = Property(Array, depends_on=['npoints'])
    soln = Property(Array, depends_on=['xdat', 'solver',
                                       'rhoL', 'rhoR', 'preL', 'preR'])
    density = Property(Array, depends_on=['soln'])
    pressure = Property(Array, depends_on=['soln'])
    rhoL = Range(0.0, 10.0)
    rhoR = Range(0.0, 10.0)
    preL = Range(0.0, 10.0)
    preR = Range(0.0, 10.0)
    npoints = Range(10, 2000, value=50)
    plot_type = Enum("scatter", "line")
    solver = Enum("exact", "hllc", "hll")

    def _rhoL_default(self): return 1.0
    def _rhoR_default(self): return 0.125
    def _preL_default(self): return 1.0
    def _preR_default(self): return 0.1

    def _get_xdat(self):
        return np.linspace(-2, 2, self.npoints)

    def _get_soln(self):
        descr = pyfluids.FluidDescriptor()
        SL = pyfluids.FluidState(descr)
        SR = pyfluids.FluidState(descr)

        SL.primitive = np.array([self.rhoL, self.preL, 0, 0, 0])
        SR.primitive = np.array([self.rhoR, self.preR, 0, 0, 0])

        solver = pyfluids.RiemannSolver()
        solver.solver = self.solver
        solver.set_states(SL, SR)
        y = np.array([solver.sample(xi).primitive for xi in self.xdat])
        return y

    def _get_density(self):
        return self.soln[:,0]
    
    def _get_pressure(self):
        return self.soln[:,1]

    def _plot_type_changed(self):
        if self.plot_type == "line":
            self.npoints = 2000
        else:
            self.npoints = 50
        
def make_plot_item(ydata, color="blue"):
    return ChacoPlotItem("xdat", ydata,
                  type_trait="plot_type",
                  resizable=True,
                  x_label="x/t",
                  y_label=ydata,
                  color=color,
                  bgcolor="white",
                  border_visible=True,
                  border_width=2,
                  show_label=False,
                  title=ydata + " vs x/t",
                  padding_bg_color="lightgray")

grp1 = Group(make_plot_item("density", color="blue"),
             make_plot_item("pressure", color="red"), show_border=True)
grp2 = HGroup(Group(Item('rhoL', label='density left', width=260),
                    Item('preL', label='pressure left', width=260)),
              Group(Item('rhoR', label='density right', width=260),
                    Item('preR', label='pressure right', width=260)))
grp3 = HGroup(VGroup(Item(name='solver', style='custom'),
                     Item(name='plot_type', style='custom', label='plot type')),
              Item(name='npoints', label="number of points", width=100))

view = View(VGroup(grp1, VGroup(grp2, grp3), show_border=True),
            width=800, height=800,
            title="Exact Riemann Solver for the Euler Equations", resizable=True)


if __name__ == "__main__":
    app = RiemannApp()
    app.configure_traits(view=view)
