import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd

T = pd.read_pickle(r'C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-WHEC\364981b5-b118-43ff-9258-81b3eebba6e8\scratchpad\imu_1s.pkl')
asc = T[(T.rel_min > 10) & (T.rel_min < 96)]
dsc = T[(T.rel_min > 99) & (T.rel_min < 160)]
print('ascent (rel 10-96 min): n=%d s' % len(asc))
for q in [50, 75, 90, 95, 99]:
    print('  tilt p%02d: %.2f deg' % (q, np.percentile(asc.tilt_deg.dropna(), q)))
print('  gyro rms median: %.2f rad/s (%.1f rpm)' % (
    asc.gyro_rms.median(), asc.gyro_rms.median() * 60 / (2 * np.pi)))
print('descent: tilt p50 %.2f p95 %.2f deg; gyro med %.2f rad/s' % (
    np.percentile(dsc.tilt_deg.dropna(), 50), np.percentile(dsc.tilt_deg.dropna(), 95),
    dsc.gyro_rms.median()))
# cosine error of direct beam for tilt d at SZA theta: ~ tan(theta)*d_rad (worst azimuth)
for sza in [45, 60, 72]:
    d95 = np.radians(np.percentile(asc.tilt_deg.dropna(), 95))
    print('direct-beam cosine err at SZA %d for p95 tilt: %.1f%%' % (
        sza, 100 * np.tan(np.radians(sza)) * d95))
