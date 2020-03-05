import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as dates

class PromotionGraph:

    def __init__(self, data=None, title=None):

        fig = plt.figure()
        fig.suptitle(title)

        # todo: create live figure
        # https://github.com/notadamking/Stock-Trading-Visualization/blob/39ed1d4dc4ce734853f76a3256ed6de5ee963192/render/StockTradingGraph.py

    def render(self):
        print('busy rendering')
