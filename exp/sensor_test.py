"""Is the INTRINSIC growth trigger actually discriminating, or is it just firing at min_gap every time?
(Growth gaps in the live run were 800/835/805/800 against min_gap=800 -- suspicious.)
Feed the sensor three synthetic entropy traces and see whether it can tell them apart:
  A) still clearly improving   -> must NOT fire
  B) genuinely plateaued       -> MUST fire
  C) improving but noisy       -> must NOT fire (noise should not look like saturation)"""
import sys; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, growth as G
def trial(name,gen,window=150,t_stat=1.0,min_gap=800,n=3000,smooth=None):
    s=G.SaturationSensor(window=window,t_stat=t_stat,min_gap=min_gap); fires=[]; ema=None
    for i in range(n):
        e=gen(i)
        if smooth: ema=e if ema is None else (1-smooth)*ema+smooth*e; e=ema
        if s.observe(e,i): fires.append(i)
    print(f"  {name:34s} fires at {fires[:6]}{' ...' if len(fires)>6 else ''}  (n={len(fires)})")
    return fires
print("SENSOR DIAGNOSTIC (window=150, t=1.0, min_gap=800) -- current settings:")
rng=np.random.default_rng(0)
improving=lambda i: 8.0-2.0*(i/3000)+rng.normal(0,0.05)          # steadily falling
plateau  =lambda i: 5.0+rng.normal(0,0.05)                        # flat
noisy    =lambda i: 8.0-2.0*(i/3000)+rng.normal(0,0.35)           # falling, but noisy
trial("A improving (should NOT fire)",improving)
trial("B plateaued (SHOULD fire)",plateau)
trial("C improving+noisy (should NOT fire)",noisy)
print("\nWith a SMOOTHED signal and a longer window (window=400, EMA 0.02):")
trial("A improving (should NOT fire)",improving,window=400,smooth=0.02)
trial("B plateaued (SHOULD fire)",plateau,window=400,smooth=0.02)
trial("C improving+noisy (should NOT fire)",noisy,window=400,smooth=0.02)
