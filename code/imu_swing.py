"""Parse Runcam GyroFlow IMU logs: quantify payload swing/tilt, find burst & landing."""
import sys, glob, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd

DCIM = r'C:\Users\Admin\Desktop\akash jothi HAB launch\footage\DCIM'
GS = 0.00053263221      # rad/s per unit
AS = 0.00048828125      # g per unit

segs = []
for n in range(4, 31):   # flight session: 0004..0030
    f = os.path.join(DCIM, 'ThumbPW_%04d.gcsv' % n)
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f, skiprows=17, dtype=np.float64,
                    names=['t', 'rx', 'ry', 'rz', 'ax', 'ay', 'az'])
    end = os.path.getmtime(f)   # camera RTC end-of-segment (epoch s)
    d['t'] = d.t / 1000.0
    dur = d.t.iloc[-1]
    d['abs_t'] = end - dur + d.t          # camera epoch seconds
    # 1-s downsample
    d['sec'] = d.abs_t.astype(np.int64)
    g = d.groupby('sec').agg(
        gyro_rms=('rx', lambda x: 0.0),   # placeholder, computed below
        ax=('ax', 'mean'), ay=('ay', 'mean'), az=('az', 'mean'))
    # gyro magnitude rms per second
    d['gmag'] = np.sqrt(d.rx**2 + d.ry**2 + d.rz**2) * GS
    g['gyro_rms'] = d.groupby('sec').gmag.apply(lambda x: np.sqrt(np.mean(x**2)))
    # accel magnitude (g) and 1-s vector
    g['amag'] = np.sqrt(g.ax**2 + g.ay**2 + g.az**2) * AS
    # per-second accel spike (max within second)
    d['amag_i'] = np.sqrt(d.ax**2 + d.ay**2 + d.az**2) * AS
    g['amax'] = d.groupby('sec').amag_i.max()
    g['seg'] = n
    segs.append(g.reset_index())
    print('seg %02d: dur %.0fs rows %d' % (n, dur, len(d)))

T = pd.concat(segs, ignore_index=True).sort_values('sec').reset_index(drop=True)
t0 = T.sec.iloc[0]
T['rel_min'] = (T.sec - t0) / 60.0
print('\ntotal timeline: %.1f min, %d one-sec samples' % (T.rel_min.iloc[-1], len(T)))

# ---- event detection ----
# burst: largest sustained gyro activity; landing: last big accel spike
i_burst = T.gyro_rms.rolling(10, center=True).mean().idxmax()
print('\nburst candidate: rel %.1f min (gyro_rms %.1f rad/s, seg %d)' % (
    T.rel_min[i_burst], T.gyro_rms[i_burst], T.seg[i_burst]))
# landing: accel spikes after burst
after = T[T.rel_min > T.rel_min[i_burst] + 10]
cands = after[after.amax > 4.0]
if len(cands):
    i_land = cands.index[-1]
    print('landing candidate: rel %.1f min (amax %.1f g, seg %d)' % (
        T.rel_min[i_land], T.amax[i_land], T.seg[i_land]))
    print('burst->landing: %.1f min (flight data says 66)' % (
        T.rel_min[i_land] - T.rel_min[i_burst]))
# top accel spikes overall
top = T.nlargest(8, 'amax')[['rel_min', 'amax', 'seg']]
print('\ntop accel spikes:')
print(top.to_string(index=False))

# ---- tilt statistics ----
# tilt = angle between 1-s accel vector and its 120-s rolling mean direction
v = T[['ax', 'ay', 'az']].values
w = pd.DataFrame(v).rolling(121, center=True, min_periods=40).mean().values
dot = (v * w).sum(1) / (np.linalg.norm(v, axis=1) * np.linalg.norm(w, axis=1))
T['tilt_deg'] = np.degrees(np.arccos(np.clip(dot, -1, 1)))
T.to_pickle(r'C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-WHEC\364981b5-b118-43ff-9258-81b3eebba6e8\scratchpad\imu_1s.pkl')
print('\nsaved imu_1s.pkl')
