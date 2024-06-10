from matplotlib import rcParams
from utils import get_figsize
from scripts.constants import columnwidth

params = {'backend': 'ps',
          'axes.labelsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'font.size': 7.0,
          'figure.figsize': get_figsize(columnwidth=columnwidth)}

print('passed by init')
rcParams.update(params)
# print(rcParams.keys())