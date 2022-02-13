import numpy as np
import matplotlib as plt
import plotly.graph_objects as go
from evaluation import *

shift_x_test = x_shift(x_test, pad_width=11)

total_white = np.sum(x_test, axis=(1, 2, 3))

max_shift = 10

x = np.arange(-max_shift, max_shift + 1, 1)
y = np.arange(-max_shift, max_shift + 1, 1)

xm, ym = np.meshgrid(x, y)

white_ratios = np.array([[np.mean(np.sum(shift_x_test(col, row), axis=(1, 2, 3)) / total_white)
                          for row in y]
                         for col in x])
'''
plt.imshow(white_ratios, cmap='autumn')
plt.show()
'''

fig = go.Figure(data=[go.Surface(x=xm, y=ym, z=white_ratios)])
fig.update_layout(title='Loss in white pixels during shift',
                  xaxis_title="X offset",
                  yaxis_title="Y offset")
fig.show()