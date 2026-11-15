"""Extract green (p53-CFP) and pink (Mdm2-YFP) curves from Geva-Zatorsky 2006 Fig 1B.

Usage: python -I extract_curves.py <image.png> <outdir>
Calibration below is for the PMC screenshot (1552x1300). For a different
image, update FRAMES, PX_PER_H and PX_PER_UNIT from that image.
"""
import sys, os, numpy as np
from PIL import Image

# Panel frames: (left x, top y, right x, bottom y) in image pixels.
FRAMES = {'cell1': (450, 748, 670, 882), 'cell2': (685, 748, 905, 882), 'cell3': (922, 748, 1142, 882),
          'cell4': (450, 898, 670, 1032), 'cell5': (685, 898, 905, 1032), 'cell6': (922, 898, 1142, 1032)}
PX_PER_H = 169 / 24      # tick labels 0 and 24 h are 169 px apart
PX_PER_UNIT = 122.5      # y = 0 at frame bottom, y = 1 is 122.5 px above it
LEGEND = (512, 900, 655, 940)  # legend box in cell4, excluded
DT_OUT = 0.25            # output grid (h)
THRESH = 30              # minimum colour score for a curve pixel

def score(img, kind):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return g - np.maximum(r, b) if kind == 'p53' else r - g

def trace(sc, top, bottom):
    """Column-wise weighted centroid of the curve run nearest the previous point."""
    ys = np.full(sc.shape[1], np.nan); prev = None
    for c in range(sc.shape[1]):
        col = sc[top:bottom, c]; on = col > THRESH
        if not on.any(): continue
        idx = np.flatnonzero(on); runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        cents = [np.average(rr, weights=col[rr]) for rr in runs]
        k = int(np.argmin([abs(x - prev) for x in cents])) if prev is not None else int(np.argmax([col[rr].sum() for rr in runs]))
        ys[c] = cents[k] + top; prev = cents[k]
    return ys

def main(path, outdir):
    img = np.array(Image.open(path).convert('RGB')).astype(int)
    os.makedirs(outdir, exist_ok=True)
    for name, (L, T, R, B) in FRAMES.items():
        x0, x1 = L + 2, R - 1                     # skip frame lines
        out = {}
        for kind in ('p53', 'mdm2'):
            sc = score(img[:, x0:x1], kind).astype(float)
            if name == 'cell4':                   # blank the legend
                sc[LEGEND[1]:LEGEND[3], max(LEGEND[0]-x0, 0):LEGEND[2]-x0] = 0
            ys = trace(sc, T + 2, B)
            t = (np.arange(x0, x1) - L) / PX_PER_H
            v = (B - ys) / PX_PER_UNIT
            ok = np.isfinite(v)
            out[kind] = (t[ok], np.clip(v[ok], 0, 1), 1 - ok.mean())
        tmax = min(out['p53'][0].max(), out['mdm2'][0].max())
        tg = np.arange(0, tmax + 1e-9, DT_OUT)
        P = np.interp(tg, out['p53'][0], out['p53'][1]); M = np.interp(tg, out['mdm2'][0], out['mdm2'][1])
        np.savetxt(os.path.join(outdir, f'p53_{name}.csv'), np.column_stack([tg, P, M]), delimiter=',',
                   header='time_h,p53_norm,mdm2_norm', comments='', fmt='%.4f')
        print(f'{name}: {len(tg)} pts, 0-{tmax:.1f} h, gap fraction p53 {out["p53"][2]:.2f} mdm2 {out["mdm2"][2]:.2f}')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
