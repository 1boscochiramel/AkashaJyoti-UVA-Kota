"""AkashaJyoti HAB (5 Oct 2025, Kota) - full analysis for SSRN paper.

Products:
  F1 flight overview (altitude/time + T, RH)
  F2 UV irradiance vs altitude, raw + SZA-corrected, with uncertainty band
  F3 vertical optical depth profile tau_above(z) +/- sigma, Rayleigh split
  F4 closure: balloon-derived AOD vs MAIAC (Angstrom-extrapolated) + buffers
  F5 context: CPCB PM launch day + Geiger profile (Pfotzer)
  results.json with all headline numbers
"""
import sys, json, re, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'C:\Users\Admin\Desktop\All data'
OUT = BASE + r'\ssrn_paper'
import os
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3,
                     'figure.dpi': 150, 'savefig.bbox': 'tight'})

R = {}  # results dict

# ================= load flight =================
df = pd.read_csv(BASE + r'\datalog_cleaned_with_altitude.csv')

# GPS time strings are IST (firmware adds +5:30; cold-start row shows 2000-0-0 5:30)
def parse_gps(s):
    m = re.search(r'(\d{4})-(\d+)-(\d+) (\d+):(\d+)', str(s))
    if not m:
        return np.nan
    y, mo, d, h, mi = map(int, m.groups())
    if y < 2020:
        return np.nan
    return h + mi / 60.0  # fractional IST hour

df['gps_h'] = df.gps_time_str.map(parse_gps)
# GPS time strings freeze at altitude; the logger RTC ticks steadily.
# Anchor RTC to the first valid GPS(IST) reading and use RTC cadence throughout.
rtc = pd.to_datetime(df.datetime)
rtc_h = (rtc - rtc.iloc[0]).dt.total_seconds() / 3600.0
anchor = df.gps_h.dropna().iloc[0]
df['ist_h'] = anchor + rtc_h
df = df.dropna(subset=['ist_h']).reset_index(drop=True)

# ================= solar zenith angle (NOAA) =================
N = 278  # day of year for 2025-10-05
lat = np.radians(df.lat.values)
lon = df.lon.values
utc_h = df.ist_h.values - 5.5
g = 2 * np.pi / 365 * (N - 1 + (utc_h - 12) / 24)
decl = (0.006918 - 0.399912*np.cos(g) + 0.070257*np.sin(g) - 0.006758*np.cos(2*g)
        + 0.000907*np.sin(2*g) - 0.002697*np.cos(3*g) + 0.00148*np.sin(3*g))
eqt = 229.18 * (0.000075 + 0.001868*np.cos(g) - 0.032077*np.sin(g)
                - 0.014615*np.cos(2*g) - 0.040849*np.sin(2*g))
tst = df.ist_h.values * 60 + eqt + 4 * lon - 60 * 5.5
ha = np.radians(tst / 4 - 180)
cossza = np.sin(lat)*np.sin(decl) + np.cos(lat)*np.cos(decl)*np.cos(ha)
df['cossza'] = cossza
df['sza_deg'] = np.degrees(np.arccos(cossza))
# Kasten-Young (1989) relative airmass, valid to the horizon
df['am'] = 1.0 / (cossza + 0.50572 * (96.07995 - df.sza_deg) ** -1.6364)
R['sza_start'] = float(df.sza_deg.iloc[0]); R['sza_end'] = float(df.sza_deg.iloc[-1])
R['t_start'] = float(df.ist_h.iloc[0]); R['t_end'] = float(df.ist_h.iloc[-1])
R['alt_min'] = float(df.alt_m.min()); R['alt_max'] = float(df.alt_m.max())

# ================= UV irradiance & per-sample errors =================
CCAL = 40.0            # W m-2 per V (Apogee SU-202)
U_CAL, U_LIN, U_REP, U_COS, U_TEMPC = 0.05, 0.01, 0.005, 0.02, 0.001
SIG_QV = 0.01 / np.sqrt(12)

df['E'] = CCAL * df.su202
rel_rand = np.sqrt(U_REP**2 + U_COS**2 + (SIG_QV / df.su202.clip(lower=1e-9))**2)
rel_sys = np.sqrt(U_CAL**2 + U_LIN**2 + (U_TEMPC * (df.temp_C - 25).abs())**2)
df['relE_rand'] = rel_rand
df['relE_tot'] = np.sqrt(rel_rand**2 + rel_sys**2)
R['E_min'] = float(df.E.min()); R['E_max'] = float(df.E.max())
R['relE_med_pct'] = float(100 * df.relE_tot.median())
R['relE_rand_med_pct'] = float(100 * df.relE_rand.median())

# ascent only (up to apogee), valid signal
iapo = int(df.alt_m.idxmax())
R['apogee_ist_h'] = float(df.ist_h[iapo])
R['ascent_rate_ms'] = float(df.alt_m[iapo] / ((df.ist_h[iapo] - df.ist_h[0]) * 3600))
asc = df.iloc[:iapo + 1]
asc = asc[asc.su202 > 0.03].copy().sort_values('alt_m')

# ================= bin profile (500 m) =================
# tau retrieval: Kasten-Young airmass valid to horizon; keep SZA <= 80,
# inflate cosine-response error to 5% at SZA > 72 (Apogee: +/-5% at 75 deg)
SZA_MAX = 80.0
ret = asc[asc.sza_deg <= SZA_MAX]
R['sza_cut'] = SZA_MAX
R['z_lowest_valid_km'] = float(ret.alt_m.min() / 1000)
zb = np.arange(0, ret.alt_m.max() + 500, 500.0)
gi = np.digitize(ret.alt_m, zb)
gr = ret.groupby(gi).agg(z=('alt_m', 'mean'), V=('su202', 'mean'), sdV=('su202', 'std'),
                         n=('su202', 'size'), cs=('cossza', 'mean'), am=('am', 'mean'),
                         sza=('sza_deg', 'mean'), P=('pressure_hPa', 'mean'),
                         T=('temp_C', 'mean'), rh=('humidity_pct', 'mean'),
                         relr=('relE_rand', 'mean'))
gr = gr[gr.n >= 3].reset_index(drop=True)
E = CCAL * gr.V.values
z = gr.z.values
cs = gr.cs.values
sec = gr.am.values
ucos_bin = np.where(gr.sza.values <= 72, U_COS, 0.05)
# random error of bin-mean ln E
sig_lnE = np.sqrt((np.maximum(gr.sdV.values, SIG_QV) / np.sqrt(gr.n.values) / gr.V.values)**2
                  + U_REP**2 + ucos_bin**2)

# ================= Langley-anchored vertical optical depth =================
# ln E = ln E0 + ln cos(sza) - sec(sza) * tau_above(z)
# anchor: tau_above(ceiling) = Rayleigh above P_ceil + stratospheric aerosol
LAM_EFF = 0.360  # um, effective wavelength of SU-202 x solar spectrum
TAU_R0 = 0.0088 * LAM_EFF ** -4.05  # Rayleigh at surface (1013 hPa)
P0 = 1013.25
P_ceil = gr.P.values[-1]
TAU_RESID = TAU_R0 * P_ceil / P0 + 0.005
lnE0 = np.log(E[-1]) - np.log(cs[-1]) + sec[-1] * TAU_RESID
tau_above = (lnE0 + np.log(cs) - np.log(E)) / sec
# error: calibration cancels (common factor in E and anchor); combine random terms
sig_tau = np.sqrt(sig_lnE**2 + sig_lnE[-1]**2) / sec
sig_tau[-1] = sig_lnE[-1] / sec[-1]

tau_col = tau_above[0]
sig_tau_col = sig_tau[0]
tau_R = TAU_R0 * gr.P.values / P0          # Rayleigh above z  (prop. to P)
tau_aer = tau_above - tau_R
aod_balloon = tau_aer[0]
sig_aod_balloon = np.sqrt(sig_tau_col**2 + (0.05 * tau_R[0])**2)  # 5% on Rayleigh model

R['tau_R0'] = float(TAU_R0); R['tau_resid_anchor'] = float(TAU_RESID)
R['tau_col'] = float(tau_col); R['sig_tau_col'] = float(sig_tau_col)
R['tau_R_surf'] = float(tau_R[0])
R['aod_balloon_360'] = float(aod_balloon); R['sig_aod_balloon'] = float(sig_aod_balloon)

# ================= MAIAC closure =================
AOD550, AOD470 = 0.610, 0.781
alpha = np.log(AOD470 / AOD550) / np.log(550 / 470)
aod_maiac_360 = AOD550 * (0.550 / LAM_EFF) ** alpha
# MAIAC uncertainty over land ~ +/-(0.05 + 0.1*AOD); extrapolation adds ~10%
sig_maiac = np.sqrt((0.05 + 0.10 * aod_maiac_360)**2 + (0.10 * aod_maiac_360)**2)
R['angstrom'] = float(alpha); R['aod_maiac_550'] = AOD550; R['aod_maiac_470'] = AOD470
R['aod_maiac_360'] = float(aod_maiac_360); R['sig_maiac_360'] = float(sig_maiac)
# direct-beam extinction expectation vs measured GLOBAL-irradiance effective tau
tau_ext = TAU_R0 + aod_maiac_360
sig_tau_ext = sig_maiac
R['tau_ext_expected'] = float(tau_ext); R['sig_tau_ext'] = float(sig_tau_ext)
R['eta_global_direct'] = float(tau_col / tau_ext)
R['sig_eta'] = float(R['eta_global_direct'] *
                     math.sqrt((sig_tau_col / tau_col)**2 + (sig_tau_ext / tau_ext)**2))
R['E0_implied'] = float(math.exp(lnE0))
R['transmittance_global'] = float(math.exp(-tau_col))

# ================= CPCB launch day =================
def cpcb(fname):
    raw = pd.read_excel(BASE + '\\' + fname, header=None, skiprows=16)
    raw.columns = ['from', 'to', 'PM25', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2'] + \
                  [f'x{i}' for i in range(raw.shape[1] - 10)]
    raw['from'] = pd.to_datetime(raw['from'], format='%d-%m-%Y %H:%M', errors='coerce')
    raw = raw.dropna(subset=['from'])
    for c in ['PM25', 'PM10', 'NO2', 'SO2', 'CO', 'NH3']:
        raw[c] = pd.to_numeric(raw[c], errors='coerce')
    return raw

nay = cpcb('Nayapura.xlsx')
shr = cpcb('Shrinath_Puram.xlsx')
day = lambda d: d[(d['from'] >= '2025-10-05') & (d['from'] < '2025-10-06')]
nd, sd = day(nay), day(shr)
R['nay_pm25_day'] = float(nd.PM25.mean()); R['nay_pm10_day'] = float(nd.PM10.mean())
R['shr_pm25_day'] = float(sd.PM25.mean()); R['shr_pm10_day'] = float(sd.PM10.mean())
# flight window 8:00-12:00 bin
fw = lambda d: d[(d['from'].dt.day == 5) & (d['from'].dt.month == 10) & (d['from'].dt.hour == 8)]
R['nay_pm25_flight'] = float(fw(nd).PM25.iloc[0]) if len(fw(nd)) else None
R['shr_pm25_flight'] = float(fw(sd).PM25.iloc[0]) if len(fw(sd)) else None

# ================= Geiger (raw datalog) =================
rawlog = open(r'C:\Users\Admin\Desktop\akash jothi HAB launch\Sensor Data\datalog.csv',
              encoding='utf-8', errors='ignore').read().splitlines()
cpm_t, cpm_v, pres_v = [], [], []
for line in rawlog:
    parts = line.split(',')
    if len(parts) < 11:
        continue
    try:
        cpm = float(parts[9].replace('CPM :', ''))
        pres = float(parts[8].replace('Pressure', ''))
        cpm_t.append(parts[0]); cpm_v.append(cpm); pres_v.append(pres)
    except Exception:
        continue
gg = pd.DataFrame({'cpm': cpm_v, 'P': pres_v})
gg['alt'] = 44330 * (1 - (gg.P / P0) ** 0.1903)
gg = gg.iloc[:gg.alt.idxmax() + 1]          # ascent only
# tube HV fails at low pressure (corona discharge -> 1e5 CPM, then 0);
# keep the physically valid segment below the failure
gg = gg[(gg.cpm >= 0) & (gg.cpm < 500) & (gg.alt < 4700)]
gg['cpm_s'] = gg.cpm.rolling(15, center=True, min_periods=5).median()
R['cpm_ground'] = float(gg[gg.alt < 500].cpm.mean())
top = gg[gg.alt > 4000]
R['cpm_4km'] = float(top.cpm.median()) if len(top) else None
R['geiger_fail_km'] = 4.7

# ================= AOD buffers =================
aodstat = pd.read_csv(BASE + r'\AOD_Stat.csv')
aodstat['Wavelength (nm)'] = aodstat['Wavelength (nm)'].ffill()

# ================= TROPOMI along-track columns (context) =================
s5p = pd.read_csv(BASE + r'\Satellite_data_extracted.csv')
DU = 4.4615e-4  # mol m-2 per Dobson unit
R['o3_DU'] = float(s5p.O3_October_2025.mean() / DU)
R['no2_umolm2'] = float(s5p.NO2_October_2025.mean() * 1e6)
R['so2_umolm2'] = float(s5p.SO2_October_2025.mean() * 1e6)
R['co_molm2'] = float(s5p.CO_October_2025.mean())
R['uvai_mean'] = float(s5p.UVAI_October_2025.mean())

# ================= FIGURES =================
# F1 overview
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].plot(df.ist_h, df.alt_m / 1000, lw=1.2, color='navy')
ax[0].set_xlabel('Local time IST (h)'); ax[0].set_ylabel('Altitude (km)')
ax[0].set_title('(a) Flight profile')
ax2 = ax[1]
ax2.plot(df.temp_C, df.alt_m / 1000, lw=1, color='crimson', label='T (°C)')
ax2.plot(df.humidity_pct, df.alt_m / 1000, lw=1, color='teal', label='RH (%)')
ax2.set_xlabel('T (°C) / RH (%)'); ax2.set_ylabel('Altitude (km)')
ax2.legend(); ax2.set_title('(b) Thermodynamic profile')
fig.savefig(OUT + r'\F1_overview.pdf'); fig.savefig(OUT + r'\F1_overview.png'); plt.close(fig)

# F2 UV vs altitude
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
ax[0].plot(asc.E, asc.alt_m / 1000, '.', ms=2, alpha=0.35, color='gray', label='samples')
ax[0].errorbar(E, z / 1000, xerr=E * gr.relr.values, fmt='o', ms=4, color='navy',
               lw=1, label='500 m bin mean ±σ')
ax[0].set_xlabel('UV-A irradiance E (W m$^{-2}$)'); ax[0].set_ylabel('Altitude (km)')
ax[0].legend(); ax[0].set_title('(a) Measured')
En = E / cs
ax[1].errorbar(En, z / 1000, xerr=En * sig_lnE, fmt='o', ms=4, color='darkgreen', lw=1)
ax[1].set_xlabel('E / cos($\\theta_s$) (W m$^{-2}$)')
ax[1].set_title('(b) Geometry-normalised')
fig.savefig(OUT + r'\F2_uv_profile.pdf'); fig.savefig(OUT + r'\F2_uv_profile.png'); plt.close(fig)

# F3 tau profile
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot(tau_above, z / 1000, '-o', ms=4, color='navy', label='$\\tau_{above}(z)$ (total)')
ax.fill_betweenx(z / 1000, tau_above - sig_tau, tau_above + sig_tau, color='navy', alpha=0.18)
ax.plot(tau_R, z / 1000, '--', color='crimson', label='Rayleigh ($\\propto P$)')
ax.plot(tau_aer, z / 1000, '-s', ms=3, color='darkorange', label='aerosol residual')
ax.set_xlabel('Vertical optical depth above z (355–365 nm eff.)')
ax.set_ylabel('Altitude (km)'); ax.legend()
fig.savefig(OUT + r'\F3_tau_profile.pdf'); fig.savefig(OUT + r'\F3_tau_profile.png'); plt.close(fig)

# F4 global vs direct-beam optical depth
fig, ax = plt.subplots(figsize=(5.8, 4))
items = [('$\\tau_{eff}$ balloon\n(global UV-A)', tau_col, sig_tau_col, 'navy'),
         ('Rayleigh\n360 nm', TAU_R0, 0.05 * TAU_R0, 'crimson'),
         ('MAIAC AOD\n470 nm', AOD470, 0.05 + 0.1 * AOD470, 'gray'),
         ('MAIAC AOD\n550 nm', AOD550, 0.05 + 0.1 * AOD550, 'gray'),
         ('MAIAC AOD\n→360 nm', aod_maiac_360, sig_maiac, 'darkorange'),
         ('$\\tau_{ext}$ direct-beam\nexpectation', tau_ext, sig_tau_ext, 'black')]
for i, (lab, v, e, c) in enumerate(items):
    ax.errorbar(i, v, yerr=e, fmt='o', ms=8, color=c, capsize=5)
ax.set_xticks(range(len(items)))
ax.set_xticklabels([i[0] for i in items], fontsize=7)
ax.set_ylabel('Optical depth')
ax.set_title('Global-irradiance vs direct-beam optical depth, 5 Oct 2025')
ax.annotate('$\\eta = \\tau_{eff}/\\tau_{ext}$ = %.3f ± %.3f' %
            (R['eta_global_direct'], R['sig_eta']),
            xy=(0.03, 0.9), xycoords='axes fraction', fontsize=9)
fig.savefig(OUT + r'\F4_closure.pdf'); fig.savefig(OUT + r'\F4_closure.png'); plt.close(fig)

# F5 context
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for d, lab, c in [(nay, 'Nayapura', 'navy'), (shr, 'Shrinath Puram', 'darkorange')]:
    w = d[(d['from'] >= '2025-10-03') & (d['from'] < '2025-10-08')]
    ax[0].plot(w['from'], w.PM25, '-o', ms=3, lw=1, color=c, label=lab + ' PM$_{2.5}$')
ax[0].axvspan(pd.Timestamp('2025-10-05 07:14'), pd.Timestamp('2025-10-05 10:30'),
              color='gold', alpha=0.35, label='flight')
ax[0].set_ylabel('PM$_{2.5}$ (µg m$^{-3}$)'); ax[0].legend(fontsize=7)
ax[0].tick_params(axis='x', rotation=45, labelsize=7)
ax[0].set_title('(a) CPCB surface PM$_{2.5}$')
ax[1].plot(gg.cpm_s, gg.alt / 1000, lw=1.2, color='purple')
ax[1].fill_betweenx(gg.alt / 1000, gg.cpm_s - np.sqrt(gg.cpm_s.clip(lower=1)),
                    gg.cpm_s + np.sqrt(gg.cpm_s.clip(lower=1)), color='purple', alpha=0.15)
ax[1].axhline(4.7, ls='--', color='gray')
ax[1].text(0.05, 0.78, 'GM tube HV failure\nabove ≈4.7 km (excluded)',
           transform=ax[1].transAxes, fontsize=8)
ax[1].set_xlabel('Count rate (CPM, 15-pt median ±$\\sqrt{N}$)')
ax[1].set_ylabel('Altitude (km)')
ax[1].set_title('(b) Cosmic-ray count profile, valid segment')
fig.savefig(OUT + r'\F5_context.pdf'); fig.savefig(OUT + r'\F5_context.png'); plt.close(fig)

json.dump(R, open(OUT + r'\results.json', 'w'), indent=1)
print(json.dumps(R, indent=1))
print('\nAOD buffer stats (550nm):')
print(aodstat[aodstat['Wavelength (nm)'] == 550].to_string(index=False))
print('\nfigures + results.json ->', OUT)
