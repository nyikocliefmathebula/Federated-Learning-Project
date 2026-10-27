# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Figure 3.1: experimental workflow diagram, drawn as SVG and converted to PDF and PNG.

Run from the project root:  python src/report/diagrams/fig3_1_workflow.py
Output: results/report/figures/ch3_1_workflow.svg, .pdf and .png (the report uses the PNG)
Needs: pip install cairosvg (and the system library libcairo2). The Inter font is used if installed.
Edit the words in the box(...) and text(...) calls below to change a label.
"""
from pathlib import Path

import cairosvg

W, H = 1200, 1090
INK, SUB = '#1d2230', '#454b5c'
ARROW = '#3a3f4b'
PROC = ('#f2f3fd', '#9097c4')      # processing steps
MODEL = ('#cfe1fb', '#5b8fd6')     # training approaches
END = ('#36b6f2', '#1a86c2')       # input datasets
FINAL = ('#7fd16a', '#3f9a2c')     # final comparison
out = []


def text(x, y, t, size=14, weight='400', style='normal', color=SUB, anchor='middle'):
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" font-style="{style}" '
               f'fill="{color}" text-anchor="{anchor}">{t}</text>')


def box(x, y, w, h, fill, title, lines=(), ts=17, ls=14, sub_italic=False):
    f, s = fill
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{f}" stroke="{s}" stroke-width="1.6"/>')
    n = 1 + len(lines)
    step = ls + 7
    y0 = y + h / 2 - (n - 1) * step / 2 + ts * 0.35
    text(x + w / 2, y0, title, ts, '700', color=INK)
    for i, t in enumerate(lines):
        text(x + w / 2, y0 + (i + 1) * step, t, ls, style='italic' if sub_italic else 'normal')


def line(pts, head=True):
    p = ' '.join(f'{x},{y}' for x, y in pts)
    m = ' marker-end="url(#ah)"' if head else ''
    out.append(f'<polyline points="{p}" fill="none" stroke="{ARROW}" stroke-width="2"{m}/>')


M = 60
FW = W - 2 * M
CX = W / 2
SEP = '\u2003|\u2003'

# input datasets
box(CX - 230, 28, 460, 74, END, f'CIC-IoT2023{SEP}CIC-IDS2017', ['Run separately'], sub_italic=True)
line([(CX, 102), (CX, 133)])
box(M, 136, FW, 58, PROC, 'Audit, clean and map labels')
line([(CX, 194), (CX, 247)])

# repeated block
fy0, fy1 = 220, 958
out.append(f'<rect x="{M - 22}" y="{fy0}" width="{FW + 44}" height="{fy1 - fy0}" rx="12" fill="none" '
           f'stroke="#5f6473" stroke-width="1.4" stroke-dasharray="6 5"/>')
out.append(f'<rect x="{M - 8}" y="{fy0 - 11}" width="168" height="22" fill="white"/>')
text(M, fy0 + 5, 'Repeated five times', 14, '700', 'italic', INK, 'start')

box(M, 250, FW, 72, PROC, 'Create a leakage-resistant split',
    [f'70% training{SEP}10% validation{SEP}20% test'])
line([(CX, 322), (CX, 351)])
box(M, 354, FW, 72, PROC, 'Fit preprocessing on training data only',
    ['Apply the frozen transformation to validation &amp; test data'])

# branch: pooled vs partitioned
G = 24
mw = (FW - 3 * G) / 4
mx = [M + i * (mw + G) for i in range(4)]
pool_c = mx[0] + mw / 2
part_x0, part_x1 = mx[1], M + FW
part_c = (part_x0 + part_x1) / 2
by = 458
line([(CX, 426), (CX, by)], head=False)
line([(pool_c, by), (part_c, by)], head=False)
line([(pool_c, by), (pool_c, 487)])
line([(part_c, by), (part_c, 487)])
box(mx[0], 490, mw, 72, PROC, 'Pooled training set', ['No client partition'])
box(part_x0, 490, part_x1 - part_x0, 72, PROC, 'Create matched client partitions',
    ['Dirichlet α ∈ {100, 10, 1, 0.5, 0.1}'])

# training approaches
my = 612
line([(pool_c, 562), (pool_c, my - 3)])
cy = 588
line([(part_c, 562), (part_c, cy)], head=False)
line([(mx[1] + mw / 2, cy), (mx[3] + mw / 2, cy)], head=False)
for i in (1, 2, 3):
    line([(mx[i] + mw / 2, cy), (mx[i] + mw / 2, my - 3)])
names = [('Centralized', 'One shared model'), ('Local-only', 'One model per gateway'),
         ('FedAvg', 'One shared model'), ('FedProx', 'One shared model')]
for i, (t, s) in enumerate(names):
    box(mx[i], my, mw, 72, MODEL, t, [s])

# converge to checkpoint selection
jy = 712
for i in range(4):
    line([(mx[i] + mw / 2, my + 72), (mx[i] + mw / 2, jy)], head=False)
line([(mx[0] + mw / 2, jy), (mx[3] + mw / 2, jy)], head=False)
line([(CX, jy), (CX, 735)])
box(M, 738, FW, 72, PROC, 'Validation-based checkpoint selection',
    [f'Shared models: global validation{SEP}Local models: gateway-specific validation'])
line([(CX, 810), (CX, 835)])

# held-out evaluation with three views
ey, eh = 838, 104
out.append(f'<rect x="{M}" y="{ey}" width="{FW}" height="{eh}" rx="8" fill="{PROC[0]}" stroke="{PROC[1]}" stroke-width="1.6"/>')
text(CX, ey + 26, 'Held-out test evaluation', 17, '700', color=INK)
vw = (FW - 4 * 16) / 3
views = [('Global threat-landscape evaluation', 'Global test traffic'),
         ('Gateway-matched evaluation', 'Each gateway’s test traffic'),
         ('Routed deployment evaluation', 'Traffic routed to local models')]
for i, (t, s) in enumerate(views):
    vx = M + 16 + i * (vw + 16)
    out.append(f'<rect x="{vx}" y="{ey + 40}" width="{vw}" height="52" rx="6" fill="white" stroke="#b9bedc" stroke-width="1.2"/>')
    text(vx + vw / 2, ey + 61, t, 14.5, '700', 'italic', INK)
    text(vx + vw / 2, ey + 81, s, 13.5)
line([(CX, ey + eh), (CX, 987)])

# final comparison
box(M, 990, FW, 76, FINAL, 'Compare results',
    ['Accuracy, macro-F1, per-class recall, binary FPR, convergence and communication cost in rounds &amp; bytes'])

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Inter, Helvetica, Arial, sans-serif">'
       '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       f'<path d="M0,0 L10,5 L0,10 z" fill="{ARROW}"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="white"/>' + ''.join(out) + '</svg>')
# Inter has no element-of sign, so this one line uses DejaVu Sans
svg = svg.replace('text-anchor="middle">Dirichlet', 'text-anchor="middle" font-family="DejaVu Sans">Dirichlet')
OUT = Path(__file__).resolve().parents[3] / 'results' / 'report' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
base = str(OUT / 'ch3_1_workflow')
open(base + '.svg', 'w').write(svg)
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=base + '.pdf')
cairosvg.svg2png(bytestring=svg.encode(), write_to=base + '.png', output_width=W * 2)
print('FIGURE_OK', base + '.png')
