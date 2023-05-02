import visdom
import numpy as np
cfg = {"server": "localhost",
       "port": 8097}

vis = visdom.Visdom('http://' + cfg["server"], port = cfg["port"])
vis.text('Hello Visdom!')
image = np.random.randint(0, 255, (3, 256, 256), dtype=np.uint8)
vis.image(image)

from visdom import Visdom
#
# class VisdomLinePlotter(object):
#     """Plots to Visdom"""
#     def __init__(self, env_name='main'):
#         self.viz = Visdom()
#         self.env = env_name
#         self.plots = {}
#     def plot(self, var_name, split_name, title_name, x, y):
#         if var_name not in self.plots:
#             self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
#                 legend=[split_name],
#                 title=title_name,
#                 xlabel='Epochs',
#                 ylabel=var_name
#             ))
#         else:
#             self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

import visdom
print(dir(visdom))