from data.std_lib import *
import data.global_vars as gv

import data.utils as func

gv.folder = 'ChRM04'
func.get_frame_rate()
print(gv.frame_rate)

func.get_delays_times()
print(gv.t_early_delay)

gv.folder = ''
func.get_frame_rate()
print(gv.frame_rate)

func.get_delays_times()
print(gv.t_early_delay)

print(np.sqrt(2))
