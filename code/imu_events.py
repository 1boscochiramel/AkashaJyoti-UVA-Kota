import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd

T = pd.read_pickle(r'C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-WHEC\364981b5-b118-43ff-9258-81b3eebba6e8\scratchpad\imu_1s.pkl')

# 2-min block summary of gyro activity
T['blk'] = (T.rel_min // 2).astype(int) * 2
b = T.groupby('blk').agg(g_med=('gyro_rms', 'median'), g_p90=('gyro_rms', lambda x: x.quantile(0.9)),
                         tilt_med=('tilt_deg', 'median'), amax=('amax', 'max'))
pd.set_option('display.max_rows', 200)
print(b.round(2).to_string())
