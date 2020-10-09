import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage.filters import gaussian_filter1d

import warnings
warnings.filterwarnings("ignore")

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
