import numpy as np
import time
import ctypes
import pdb
from ctypes import *

"""
gcc network_evaluate.c -fpic -shared -o network_evaluate.so
"""
class control_t_n(Structure):
    _fields_ = [("w0",c_float),
                ("w1",c_float),
                ("w2",c_float),
                ("w3",c_float),
                ("w4",c_float),
                ("w5",c_float),
                ("zita0",c_float),
                ("zita1",c_float),
                ("zita2",c_float),
                ("zita3",c_float),
                ("zita4",c_float),
                ("zita5",c_float),
                ]


class control_soft():
    def __init__(self,):
        self.control_n = control_t_n()
        self.control_n.w0 = 0.0
        self.control_n.w1 = 1.0
        self.control_n.w2 = 2.0
        self.control_n.w3 = 3.0
        self.control_n.w4 = 2.0
        self.control_n.w5 = 3.0
        self.control_n.zita0 = 0.1
        self.control_n.zita1 = 0.1
        self.control_n.zita2 = 0.1
        self.control_n.zita3 = 0.1
        self.control_n.zita4 = 0.1
        self.control_n.zita5 = 0.1

        self.network = cdll.LoadLibrary('/home/fyt/project/quad_raisim/models/0715_model/network_evaluate.so')
        PARAM = c_float * 18
        self.state = PARAM()
        self.network.networkEvaluate.restype = control_t_n


    def step(self, state):
        for i in range(len(state)):
            self.state[i] = state[i]
        self.control = self.network.networkEvaluate(byref(self.control_n), byref(self.state))
        return np.array([self.control.w0 ,
                        self.control.w1 ,
                        self.control.w2 ,
                        self.control.w3 ,
                        self.control.w4 ,
                        self.control.w5 ,
                        self.control.zita0,
                        self.control.zita1 ,
                        self.control.zita2 ,
                        self.control.zita3 ,
                        self.control.zita4 ,
                        self.control.zita5
                        ])

if __name__ == '__main__':
    policy = control_soft()
    s = "-6.87651167e-03  1.32076288e-02 -2.41888843e+00 -6.83277407e-02 \
  1.46175142e-02  2.61920995e+00  9.99996958e-01 -2.46152826e-03 \
  1.56852017e-04  2.46392747e-03  9.99838970e-01 -1.77753671e-02 \
 -1.13072191e-04  1.77756995e-02  9.99841993e-01  2.02919670e-02 \
  2.71129106e-01  7.00618458e-03 "
#thrust_cmds: [0.36754746 0.70599647 0.32788946 0.68492618"
#[j  for j in res if len(j)!=0]
    tmp = [i for i in s.split(" ") if len(i)!=0]
    state = np.array([float(i) for i in tmp])
    # state = np.array([-0.61168093, -1.28471716, -0.65060847,  0.04038335,  0.05663792,
    #    -0.11536029, -0.93220889,  0.36188721, -0.00492293, -0.36190093,
    #    -0.93221389,  0.00223051, -0.00378203,  0.00386091,  0.99998539,
    #     0.52255388,  0.51655515,  0.13439751])
    
    for i in range(10):
        print(state)
    #     state = np.array([ 0.32101795,  1.32658975, -1.91806674,  0.22934255, -0.09602054,
    #     0.05587601,  0.88532044, -0.39141754, -0.25099808,  0.2282965 ,
    #     0.83616555, -0.49870621,  0.4050783 ,  0.38421281,  0.8296337 ,
    #    -0.40642085, -0.83500232, -0.40642085])
        thrust = policy.step(state)
        print(thrust)