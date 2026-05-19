"""
Generate UML architecture diagrams as PNG images for the QNIM thesis.
Saves to reports/figures/thesis/
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures', 'thesis')
os.makedirs(OUT_DIR, exist_ok=True)

DPI = 180

# ──────────────────────────────────────────────────────────────────────────────
# Helper: draw a box with optional stereotype banner
# ──────────────────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, stereotype='', fc='#f0f4ff', ec='#334', lw=1.2, fontsize=9):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.04", fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(rect)
    if stereotype:
        ax.text(x, y + h/2 - 0.28, f'«{stereotype}»',
                ha='center', va='top', fontsize=fontsize - 1.5,
                fontstyle='italic', color='#555', zorder=4)
        ax.text(x, y - 0.05, label,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', zorder=4)
    else:
        ax.text(x, y, label,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', zorder=4)

def arrow(ax, x0, y0, x1, y1, style='->', color='#334', lw=1.2, label='', dashed=False):
    ls = '--' if dashed else '-'
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls),
                zorder=2)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.05, my + 0.05, label, fontsize=7.5,
                color='#444', ha='left', va='bottom', zorder=4)


# ==============================================================================
# FIGURE 1 — Hexagonal Ports-and-Adapters Architecture (includes Frontend)
# ==============================================================================
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16); ax.set_ylim(0, 9)
ax.axis('off')
ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('#fafafa')

# Domain core
box(ax, 8, 4.5, 2.8, 1.6, 'Domain\nCore', fc='#e8e8e8', ec='#222', lw=2.2, fontsize=11)

# ── Left: Driving adapters (Frontend/Input) ──
ax.text(2.1, 8.3, 'Driving Adapters  (Frontend / Input)',
        ha='center', fontsize=10, fontweight='bold', color='#c05000')

box(ax, 2.1, 7.2, 3.2, 0.9, 'TygerCLI\n(Typer)', stereotype='adapter', fc='#ffe8cc', ec='#c05000', fontsize=8.5)
box(ax, 2.1, 5.9, 3.2, 0.9, 'FastAPIRouter\n(REST API)', stereotype='adapter', fc='#ffe8cc', ec='#c05000', fontsize=8.5)
box(ax, 2.1, 4.6, 3.2, 0.9, 'HTMLDashboard\n(Jinja2)', stereotype='adapter', fc='#ffe8cc', ec='#c05000', fontsize=8.5)

# Port (left side of core)
box(ax, 5.8, 5.6, 2.2, 0.75, 'IApplicationPort', stereotype='port', fc='#d4eaf7', ec='#1a6090', fontsize=8)

# Arrows: adapters → port
for y_src in [7.2, 5.9, 4.6]:
    arrow(ax, 3.7, y_src, 5.8, 5.6, label='')
# port → core
arrow(ax, 6.9, 5.6, 7.6, 4.8)

# ── Right: Driven adapters (Infrastructure/Output) ──
ax.text(13.9, 8.3, 'Driven Adapters  (Infrastructure / Output)',
        ha='center', fontsize=10, fontweight='bold', color='#006060')

# Domain ports (right side of core)
port_ys = [7.0, 5.9, 4.8, 3.7]
port_labels = ['IQuantumClassifier', 'IQuantumAnnealer', 'IEventRepository', 'IModelRepository']
for py, pl in zip(port_ys, port_labels):
    box(ax, 10.3, py, 2.6, 0.7, pl, stereotype='port', fc='#d4eaf7', ec='#1a6090', fontsize=7.8)
    arrow(ax, 8.6 + (0 if py > 4 else 0), 4.5, 9.0, py)

# Concrete adapters
adapters = [
    (13.9, 7.3, 'QiskitVQCTrainer\n(IBM Runtime)'),
    (13.9, 6.2, 'DWaveAdapter\n(Ocean SDK)'),
    (13.9, 5.0, 'GWOSCClient\n(gwpy)'),
    (13.9, 3.8, 'SQLiteEventRepo\n(alembic)'),
    (13.9, 2.5, 'AerSimulator\n(fallback)', ),
]
for i, (ax_x, ax_y, lbl) in enumerate(adapters):
    box(ax, ax_x, ax_y, 3.0, 0.85, lbl, stereotype='adapter', fc='#ccede8', ec='#006060', fontsize=8)
    if i < 4:
        arrow(ax, 11.6, port_ys[i], 12.4, ax_y)
    else:
        arrow(ax, 11.6, port_ys[0], 12.4, ax_y, dashed=True, label='fallback')

ax.set_title('QNIM Ports-and-Adapters (Hexagonal) Architecture', fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_hexagonal_arch.png'), dpi=DPI, bbox_inches='tight')
plt.close()
print('✓ fig_hexagonal_arch.png')


# ==============================================================================
# FIGURE 2 — UML Class Diagram: Domain Model
# ==============================================================================
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18); ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('#fafafa')

ax.set_title('QNIM Domain Model — UML Class Diagram', fontsize=13, fontweight='bold', pad=8)

# Row 1: Value objects (green)
vo_color = '#e6f4e6'; vo_edge = '#2a7a2a'
vos = [(2.0, 8.5, 'Strain'), (5.5, 8.5, 'FeatureVector'),
       (9.0, 8.5, 'QuantumCircuitSpec'), (12.5, 8.5, 'QuboMatrix'), (16.0, 8.5, 'InferenceResult')]
for x, y, lbl in vos:
    box(ax, x, y, 3.0, 0.85, lbl, stereotype='value', fc=vo_color, ec=vo_edge, fontsize=8.5)

# Row 2: Entities
en_color = '#dde8ff'; en_edge = '#1a3a8a'
ents = [(3.0, 6.5, 'GravitationalWaveEvent\n[aggregate root]'),
        (9.0, 6.5, 'ExperimentRun'),
        (15.0, 6.5, 'ModelWeights')]
for x, y, lbl in ents:
    box(ax, x, y, 4.0, 0.95, lbl, stereotype='entity', fc=en_color, ec=en_edge, fontsize=8.5)

# Row 3: Port Interfaces
if_color = '#d4edf7'; if_edge = '#1a5070'
ifs = [(2.0, 4.4, 'IEventRepository'), (5.8, 4.4, 'IModelRepository'),
       (10.0, 4.4, 'IQuantumClassifier'), (14.5, 4.4, 'IQuantumAnnealer')]
for x, y, lbl in ifs:
    box(ax, x, y, 3.2, 0.8, lbl, stereotype='interface', fc=if_color, ec=if_edge, fontsize=8.5)

# Row 4: Domain Services
sv_color = '#fff9d9'; sv_edge = '#8a7000'
svs = [(2.0, 2.4, 'StrainPreprocessor'), (5.8, 2.4, 'QuboBuilder'),
       (10.0, 2.4, 'PlanckReliability\nClassifier'), (14.5, 2.4, 'MetrologyAuditor')]
for x, y, lbl in svs:
    box(ax, x, y, 3.2, 0.85, lbl, stereotype='service', fc=sv_color, ec=sv_edge, fontsize=8.5)

# Relationships
# GWE composes Strain (1..*)
arrow(ax, 3.0, 7.0, 2.0, 8.1, label='1..*\ncontains')
# ExperimentRun uses GWE and ModelWeights
arrow(ax, 7.0, 6.5, 5.0, 6.5, label='uses', dashed=True)
arrow(ax, 11.0, 6.5, 13.0, 6.5, label='uses', dashed=True)
# ExperimentRun produces InferenceResult
arrow(ax, 9.0, 6.0, 16.0, 8.1, label='produces', dashed=True)
# IEventRepository manages GWE
arrow(ax, 2.0, 4.8, 3.0, 6.0, label='manages', dashed=True)
# IModelRepository manages ModelWeights
arrow(ax, 5.8, 4.8, 14.5, 6.0, label='manages', dashed=True)
# StrainPreprocessor creates FeatureVector
arrow(ax, 2.0, 2.8, 5.5, 8.1, label='creates', dashed=True)
# QuboBuilder creates QuboMatrix
arrow(ax, 5.8, 2.8, 12.5, 8.1, label='creates', dashed=True)
# PlanckReliabilityClassifier classifies InferenceResult
arrow(ax, 10.0, 2.8, 16.0, 8.1, label='classifies', dashed=True)

# Legend
legend_items = [
    mpatches.Patch(fc=vo_color, ec=vo_edge, label='Value Object'),
    mpatches.Patch(fc=en_color, ec=en_edge, label='Entity / Aggregate'),
    mpatches.Patch(fc=if_color, ec=if_edge, label='Port Interface'),
    mpatches.Patch(fc=sv_color, ec=sv_edge, label='Domain Service'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=8.5, framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_domain_class_diagram.png'), dpi=DPI, bbox_inches='tight')
plt.close()
print('✓ fig_domain_class_diagram.png')


# ==============================================================================
# FIGURE 3 — UML Sequence Diagram: RunInferenceUseCase
# ==============================================================================
participants = [
    'Client\n(CLI/API)',
    'RunInference\nUseCase',
    'Hybrid\nOrchestrator',
    'Strain\nPreprocessor',
    'IQuantum\nClassifier',
    'IQuantum\nAnnealer',
    'Planck\nClassifier',
]
n = len(participants)
xs = np.linspace(1.0, 15.0, n)
y_top = 9.2
y_bot = 0.4

fig, ax = plt.subplots(figsize=(17, 11))
ax.set_xlim(0, 16); ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('#fafafa')
ax.set_title('UML Sequence Diagram — RunInferenceUseCase', fontsize=13, fontweight='bold', pad=8)

# Draw header boxes and lifelines
for i, (x, p) in enumerate(zip(xs, participants)):
    box(ax, x, y_top - 0.35, 1.8, 0.85, p, fc='#dde8ff', ec='#334', fontsize=8)
    ax.plot([x, x], [y_top - 0.78, y_bot], color='#999', linestyle='--', lw=1.0, zorder=1)

# Messages: (from_idx, to_idx, y, label, is_return)
messages = [
    (0, 1, 8.2,  'execute(request)',        False),
    (1, 2, 7.5,  'run(event, cfg)',          False),
    (2, 3, 6.7,  'preprocess(strain)',       False),
    (3, 2, 6.0,  'FeatureVector',            True),
    (2, 4, 5.2,  'classify(features)',       False),
    (4, 2, 4.5,  'theory probabilities',     True),
    (2, 5, 3.7,  'estimate(qubo)',           False),
    (5, 2, 3.0,  'MAP parameters',          True),
    (2, 6, 2.2,  'classify(result)',         False),
    (6, 2, 1.5,  'reliability tier',         True),
    (2, 1, 0.95, 'InferenceResult',          True),
    (1, 0, 0.5,  'InferenceResultDTO',       True),
]

for from_i, to_i, y, lbl, is_ret in messages:
    x0, x1 = xs[from_i], xs[to_i]
    color = '#1a5aaa' if not is_ret else '#888'
    ls = '-' if not is_ret else '--'
    style = '->' if not is_ret else '->'
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.4, linestyle=ls), zorder=3)
    mx = (x0 + x1) / 2
    dy = 0.12
    ax.text(mx, y + dy, lbl, ha='center', va='bottom', fontsize=7.8,
            color='#333', style='italic' if is_ret else 'normal', zorder=4)

# Activation boxes (narrow rectangles on lifeline)
act_boxes = [(1, 8.55, 0.45), (2, 7.85, 6.9), (3, 7.05, 6.35),
             (4, 5.6, 4.8), (5, 4.1, 3.35), (6, 2.6, 1.85)]
for idx, y_top_box, y_bot_box in act_boxes:
    rect = FancyBboxPatch((xs[idx]-0.12, y_bot_box), 0.24, y_top_box - y_bot_box,
                          boxstyle="square,pad=0", fc='#c8d8ff', ec='#334', lw=0.8, zorder=2)
    ax.add_patch(rect)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_inference_sequence.png'), dpi=DPI, bbox_inches='tight')
plt.close()
print('✓ fig_inference_sequence.png')


# ==============================================================================
# FIGURE 4 — ER Diagram: Persistence Model
# ==============================================================================
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16); ax.set_ylim(0, 8)
ax.axis('off')
ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('#fafafa')
ax.set_title('QNIM Persistence Model — Entity-Relationship Diagram', fontsize=13, fontweight='bold', pad=8)

er_fc = '#f0f0f0'; er_ec = '#333'
attr_fc = '#fafafa'; attr_ec = '#888'

def er_entity(ax, cx, cy, name, attrs, w=3.2):
    h_header = 0.6
    h_attr = 0.38 * len(attrs)
    total_h = h_header + h_attr
    # Header
    rect_h = FancyBboxPatch((cx - w/2, cy - h_header/2), w, h_header,
                             boxstyle="square,pad=0", fc='#c8d0e8', ec='#333', lw=1.5, zorder=3)
    ax.add_patch(rect_h)
    ax.text(cx, cy, name, ha='center', va='center', fontsize=10,
            fontweight='bold', zorder=4)
    # Attr section
    rect_a = FancyBboxPatch((cx - w/2, cy - h_header/2 - h_attr), w, h_attr,
                             boxstyle="square,pad=0", fc='#f4f4fa', ec='#888', lw=1.0, zorder=3)
    ax.add_patch(rect_a)
    for j, a in enumerate(attrs):
        ay = cy - h_header/2 - (j + 0.5) * 0.38
        ax.text(cx - w/2 + 0.15, ay, a, ha='left', va='center', fontsize=8, zorder=4)
    return cy - h_header/2 - h_attr  # bottom y

# Entities
by1 = er_entity(ax,  2.5, 5.8, 'GW_EVENT',
                ['PK  id', 'event_name', 'gps_time', 'snr', 'detector_list'])
by2 = er_entity(ax,  7.5, 5.8, 'EXPERIMENT_RUN',
                ['PK  id', 'FK  event_id', 'FK  model_ver', 'backend', 'wall_time'])
by3 = er_entity(ax, 13.0, 5.8, 'MODEL_WEIGHTS',
                ['PK  version_tag', 'sha256', 'qubit_count', 'training_date'])
by4 = er_entity(ax,  7.5, 2.2, 'INFERENCE_RESULT',
                ['PK  id', 'FK  run_id', 'theory_class', 'bayes_factor',
                 'reliability_tier', 'qfi_advantage'])

# Relationship lines with cardinality
# GW_EVENT -|------<- EXPERIMENT_RUN
ax.plot([3.9, 5.9], [5.8, 5.8], color='#333', lw=1.8, zorder=2)
ax.text(4.2, 6.0, '1', ha='center', fontsize=9, color='#444')
ax.text(5.6, 6.0, 'N', ha='center', fontsize=9, color='#444')
ax.text(4.8, 6.25, 'triggers', ha='center', fontsize=9, fontstyle='italic', color='#555')

# EXPERIMENT_RUN --->|-- MODEL_WEIGHTS
ax.plot([9.1, 11.4], [5.8, 5.8], color='#333', lw=1.8, zorder=2)
ax.text(9.4, 6.0, 'N', ha='center', fontsize=9, color='#444')
ax.text(11.1, 6.0, '1', ha='center', fontsize=9, color='#444')
ax.text(10.25, 6.25, 'uses', ha='center', fontsize=9, fontstyle='italic', color='#555')

# EXPERIMENT_RUN --|-- INFERENCE_RESULT (vertical)
run_bottom = by2
ir_top = 2.2 + 0.6/2
ax.plot([7.5, 7.5], [run_bottom, ir_top], color='#333', lw=1.8, zorder=2)
ax.text(7.7, (run_bottom + ir_top)/2 + 0.2, '1', ha='left', fontsize=9, color='#444')
ax.text(7.7, (run_bottom + ir_top)/2 - 0.1, '1', ha='left', fontsize=9, color='#444')
ax.text(8.0, (run_bottom + ir_top)/2, 'produces', ha='left', fontsize=9, fontstyle='italic', color='#555')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_er_diagram.png'), dpi=DPI, bbox_inches='tight')
plt.close()
print('✓ fig_er_diagram.png')

print('\nAll UML diagrams generated successfully.')
