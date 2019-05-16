import numpy.random as Random
import google_traces_radar
import glob

Random.seed(6)
if not len(glob.glob("results/radar_plot.csv")) > 0:
    google_traces_radar.make_datas_var_options_var_sps()

google_traces_radar.make_radar_chart()

