import numpy as np

from pygfunction.boreholes import rectangle_field
from pygfunction.gfunction import gFunction
from pygfunction.load_aggregation import ClaessonJaved

def main():
    field = rectangle_field(10,7,2,2,150,2,0.2)
    time = ClaessonJaved(3600, 3600*8760*20).get_times_for_simulation()
    g_func = gFunction(field, 1/5000/1000, time).gFunc
    assert np.all(g_func > 0)


# Main function
if __name__ == '__main__':
    main()
