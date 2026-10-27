# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Figure 4.1: software architecture diagram, drawn as SVG and converted to PDF and PNG.

Run from the project root:  python src/report/diagrams/fig4_1_software_architecture.py
Output: results/report/figures/ch4_1_software_architecture.svg, .pdf and .png (the report uses the PNG)
Needs: pip install cairosvg (and the system library libcairo2). The Inter font is used if installed.
Edit the words in the box(...) calls below to change a label.
"""
from pathlib import Path

import cairosvg

W, H = 1200, 950
INK, SUB = '#1d2230', '#454b5c'
ARROW = '#3a3f4b'
PROC = ('#f2f3fd', '#9097c4')      # processes (scripts)
STORE = ('#cdf3fb', '#38a8c2')     # folders on disk
END = ('#36b6f2', '#1a86c2')       # input and final output
FLW = ('#cfe1fb', '#5b8fd6')       # Flower components

out = []


def box(x, y, w, h, fill, title, lines=(), file=None, ts=16, ls=14, dash=None, rx=8):
    f, s = fill
    d = f' stroke-dasharray="{dash}"' if dash else ''
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{f}" stroke="{s}" stroke-width="1.6"{d}/>')
    n = 1 + len(lines) + (1 if file else 0)
    step = ls + 5
    y0 = y + h / 2 - (n - 1) * step / 2 + ts * 0.35
    cx = x + w / 2
    out.append(f'<text x="{cx}" y="{y0}" font-size="{ts}" font-weight="700" fill="{INK}" text-anchor="middle">{title}</text>')
    yy = y0 + step
    for t in lines:
        out.append(f'<text x="{cx}" y="{yy}" font-size="{ls}" fill="{SUB}" text-anchor="middle">{t}</text>')
        yy += step
    if file:
        out.append(f'<text x="{cx}" y="{yy}" font-size="{ls}" font-style="italic" fill="{SUB}" text-anchor="middle">{file}</text>')


def arrow(pts, label=None, lpos=None, lsize=13):
    p = ' '.join(f'{x},{y}' for x, y in pts)
    out.append(f'<polyline points="{p}" fill="none" stroke="{ARROW}" stroke-width="2" marker-end="url(#ah)"/>')
    if label:
        x, y = lpos
        out.append(f'<text x="{x}" y="{y}" font-size="{lsize}" fill="{INK}" text-anchor="middle">{label}</text>')


# ------------------------------------------------------------ row 1: data pipeline
M, G = 40, 26
bw = (W - 2 * M - 5 * G) / 6
r1y, r1h = 40, 92
xs = [M + i * (bw + G) for i in range(6)]
box(xs[0], r1y, bw, r1h, END, 'Raw CSV files')
stages = [('Stage 1', 'audit, clean', 'clean_data.py'), ('Stage 2', 'arrays', 'build_arrays.py'),
          ('Stage 3', 'split, scale, schedule', 'split_data.py'), ('Stage 4', 'partitions', 'partition_clients.py'),
          ('Data checks', None, 'check_data.py')]
for i, (t, l, f) in enumerate(stages, start=1):
    box(xs[i], r1y, bw, r1h, PROC, t, [l] if l else [], f, ts=15, ls=13)
for i in range(5):
    arrow([(xs[i] + bw, r1y + r1h / 2), (xs[i + 1] - 3, r1y + r1h / 2)])

# ------------------------------------------------------------ row 2: processed data
r2y, r2h = 182, 72
box(M, r2y, W - 2 * M, r2h, STORE, 'data/processed/&lt;dataset&gt;/',
    ['cleaned records, arrays, splits, scaling, initial states, schedules, partitions, manifests'])
for i in range(1, 5):                       # stages write
    cx = xs[i] + bw / 2
    arrow([(cx, r1y + r1h), (cx, r2y - 3)])
cx = xs[5] + bw / 2                          # data checks read
arrow([(cx, r2y), (cx, r1y + r1h + 3)])

# ------------------------------------------------------------ row 3: shared package
r3y, r3h = 298, 72
box(M, r3y, W - 2 * M, r3h, PROC, 'shared package flower-ids/fliids/',
    ['data loading, MLP, training loop, early stopping, metrics, byte accounting, random streams'])
arrow([(W / 2, r2y + r2h), (W / 2, r3y - 3)])

# ------------------------------------------------------------ row 4: training
r4y, r4h = 455, 92
bx, bwid = M, 228
gx, gwid = 318, 228
fx0, fx1 = 590, W - M                         # Flower group
fy0, fy1 = 412, 572
box(bx, r4y, bwid, r4h, PROC, 'Centralized &amp; Local-only', [], 'train_baseline.py')
box(gx, r4y, gwid, r4h, PROC, 'Launches one run at a time', [], 'run_grid.py', ts=15)
out.append(f'<rect x="{fx0}" y="{fy0}" width="{fx1 - fx0}" height="{fy1 - fy0}" rx="10" fill="#fafbff" '
           f'stroke="#5f6473" stroke-width="1.4" stroke-dasharray="6 5"/>')
out.append(f'<text x="{(fx0 + fx1) / 2}" y="{fy0 + 26}" font-size="15" font-weight="700" fill="{INK}" '
           f'text-anchor="middle">Flower SuperLink &amp; Simulation Runtime</text>')
sw = 220
sx, cxb = fx0 + 22, fx1 - 22 - sw
box(sx, r4y, sw, r4h, FLW, 'Flower Server', ['schedule, aggregation,', 'validation, early stopping'], ts=15, ls=13)
box(cxb, r4y, sw, r4h, FLW, 'Flower Client (105)', ['local training for', 'E = 2 epochs'], ts=15, ls=13)
mid = r4y + r4h / 2
arrow([(sx + sw, mid - 14), (cxb - 3, mid - 14)])
arrow([(cxb, mid + 14), (sx + sw + 3, mid + 14)])
cxm = (sx + sw + cxb) / 2


def sym(x, y, base, sub=None, sup=None, size=16):
    """Italic math-style symbol with optional subscript and superscript, drawn as separate texts."""
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" font-style="italic" fill="{INK}">{base}</text>')
    dx = size * 0.72
    if sup:
        out.append(f'<text x="{x + dx}" y="{y - size * 0.42}" font-size="{size * 0.66}" font-style="italic" fill="{INK}">{sup}</text>')
    if sub:
        out.append(f'<text x="{x + dx}" y="{y + size * 0.3}" font-size="{size * 0.66}" font-style="italic" fill="{INK}">{sub}</text>')


sym(cxm - 8, mid - 22, 'w', sub='t')
sym(cxm - 30, mid + 40, 'w', sub='t+1', sup='k')
out.append(f'<text x="{cxm + 1}" y="{mid + 40}" font-size="16" fill="{INK}">,</text>')
out.append(f'<text x="{cxm + 10}" y="{mid + 40}" font-size="16" font-style="italic" fill="{INK}">n</text>')
out.append(f'<text x="{cxm + 19}" y="{mid + 45}" font-size="10.6" font-style="italic" fill="{INK}">k</text>')
# shared package feeds the baselines and the Flower apps
arrow([(bx + bwid / 2, r3y + r3h), (bx + bwid / 2, r4y - 3)])
arrow([((fx0 + fx1) / 2, r3y + r3h), ((fx0 + fx1) / 2, fy0 - 3)])
# run_grid launches both
arrow([(gx, mid), (bx + bwid + 3, mid)])
arrow([(gx + gwid, mid), (sx - 3, mid)])

# ------------------------------------------------------------ row 5: run folders
r5y, r5h = 616, 72
box(M, r5y, W - 2 * M, r5h, STORE, 'results/runs/&lt;dataset&gt;/&lt;method&gt;/seed&lt;S&gt;/alpha&lt;A&gt;/', ['one folder per run'])
arrow([(bx + bwid / 2, r4y + r4h), (bx + bwid / 2, r5y - 3)])
arrow([((fx0 + fx1) / 2, fy1), ((fx0 + fx1) / 2, r5y - 3)])

# ------------------------------------------------------------ row 6: analysis scripts
r6y, r6h = 730, 78
aw = (W - 2 * M - 40) / 2
ax0, ax1 = M, M + aw + 40
box(ax0, r6y, aw, r6h, PROC, 'Summary table of all runs', [], 'collect_results.py')
box(ax1, r6y, aw, r6h, PROC, 'Every local model on every gateway’s test data', [], 'cross_gateway.py')
arrow([(ax0 + aw / 2, r5y + r5h), (ax0 + aw / 2, r6y - 3)])
arrow([(ax1 + aw / 2, r5y + r5h), (ax1 + aw / 2, r6y - 3)])

# ------------------------------------------------------------ row 7: summaries
r7y, r7h = 850, 70
sw7 = 440
s7x = (W - sw7) / 2
box(s7x, r7y, sw7, r7h, END, 'results/summary/', ['analysis'])
yb = r6y + r6h
arrow([(ax0 + aw / 2, yb), (ax0 + aw / 2, r7y + r7h / 2), (s7x - 3, r7y + r7h / 2)])
arrow([(ax1 + aw / 2, yb), (ax1 + aw / 2, r7y + r7h / 2), (s7x + sw7 + 3, r7y + r7h / 2)])

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Inter, Helvetica, Arial, sans-serif">'
       '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       f'<path d="M0,0 L10,5 L0,10 z" fill="{ARROW}"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="white"/>' + ''.join(out) + '</svg>')
OUT = Path(__file__).resolve().parents[3] / 'results' / 'report' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
base = str(OUT / 'ch4_1_software_architecture')
open(base + '.svg', 'w').write(svg)
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=base + '.pdf')
cairosvg.svg2png(bytestring=svg.encode(), write_to=base + '.png', output_width=W * 2)
print('FIGURE_OK', base + '.png')
