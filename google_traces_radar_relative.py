import numpy.random as Random
import google_traces_radar
import glob
from Model.NetworkProvider import NetworkProvider
Random.seed(6)
if not len(glob.glob("results/radar_plot_relative.csv")) > 0:
    google_traces_radar.make_datas_var_options_var_sps(filename="results/radar_plot_relative.csv", ret_func=NetworkProvider()
                                   .getInstance().getRelativeBandwidthSaving)

google_traces_radar.make_radar_chart("results/radar_plot_relative.csv")
