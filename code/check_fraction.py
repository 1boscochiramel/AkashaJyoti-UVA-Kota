"""Verify the 'three quarters below 2.5 km' claim from the current retrieval."""
import sys, re, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\Admin\Desktop\All data\datalog_cleaned_with_altitude.csv')
def parse_gps(s):
    m = re.search(r'(\d{4})-(\d+)-(\d+) (\d+):(\d+)', str(s))
    if not m: return np.nan
    y, mo, d, h, mi = map(int, m.groups())
    return np.nan if y < 2020 else h + mi/60.0
df['gps_h'] = df.gps_time_str.map(parse_gps)
rtc = pd.to_datetime(df.datetime)
df['ist_h'] = df.gps_h.dropna().iloc[0] + (rtc - rtc.iloc[0]).dt.total_seconds()/3600.0
N = 278
lat = np.radians(df.lat.values); lon = df.lon.values
utc = df.ist_h.values - 5.5
g = 2*np.pi/365*(N-1+(utc-12)/24)
decl = (0.006918-0.399912*np.cos(g)+0.070257*np.sin(g)-0.006758*np.cos(2*g)
        +0.000907*np.sin(2*g)-0.002697*np.cos(3*g)+0.00148*np.sin(3*g))
eqt = 229.18*(0.000075+0.001868*np.cos(g)-0.032077*np.sin(g)-0.014615*np.cos(2*g)-0.040849*np.sin(2*g))
tst = df.ist_h.values*60+eqt+4*lon-60*5.5
ha = np.radians(tst/4-180)
cs = np.sin(lat)*np.sin(decl)+np.cos(lat)*np.cos(decl)*np.cos(ha)
df['cossza'] = cs
df['sza'] = np.degrees(np.arccos(cs))
df['am'] = 1/(cs+0.50572*(96.07995-df.sza)**-1.6364)
iapo = int(df.alt_m.idxmax())
asc = df.iloc[:iapo+1]
asc = asc[(asc.su202 > 0.03) & (asc.sza <= 80)].sort_values('alt_m')
zb = np.arange(0, asc.alt_m.max()+500, 500.0)
gi = np.digitize(asc.alt_m, zb)
gr = asc.groupby(gi).agg(z=('alt_m','mean'), V=('su202','mean'), am=('am','mean'),
                         cs=('cossza','mean'), P=('pressure_hPa','mean'), n=('su202','size'))
gr = gr[gr.n >= 3].reset_index(drop=True)
E = 40*gr.V.values
TAU_R0 = 0.0088*0.360**-4.05
resid = TAU_R0*gr.P.values[-1]/1013.25 + 0.005
lnE0 = math.log(E[-1]) - math.log(gr.cs.values[-1]) + gr.am.values[-1]*resid
tau = (lnE0 + np.log(gr.cs.values) - np.log(E)) / gr.am.values
i25 = np.argmin(np.abs(gr.z.values - 2500))
frac = (tau[0] - tau[i25]) / (tau[0] - resid)
print('tau_col %.3f, tau_above(%.0f m) %.3f, anchor %.3f' % (tau[0], gr.z.values[i25], tau[i25], resid))
print('fraction of column effect below 2.5 km: %.0f%%' % (100*frac))
i15 = np.argmin(np.abs(gr.z.values - 1500))
print('fraction below 1.5 km: %.0f%%' % (100*(tau[0]-tau[i15])/(tau[0]-resid)))
