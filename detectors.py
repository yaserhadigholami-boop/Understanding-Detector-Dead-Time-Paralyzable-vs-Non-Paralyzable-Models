import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# -------------------------------------------------
# Parameters
# -------------------------------------------------
duration = 0.02          # seconds shown on timeline
dt = 1e-5                # time step for drawing pulses
pulse_width = 0.00015    # pulse width (s)
pulse_amp = 1.0

rate0 = 800              # counts/s
tau0 = 0.0015            # dead time (s)

rng = np.random.default_rng()

# -------------------------------------------------
# Pulse shape
# -------------------------------------------------
def make_pulse(t_axis, t0, width, amp=1.0):
    sigma = width / 5
    return amp * np.exp(-0.5 * ((t_axis - t0) / sigma) ** 2)

# -------------------------------------------------
# Generate random arrival times
# -------------------------------------------------
def generate_arrivals(rate, duration, rng):
    n = rng.poisson(rate * duration)
    arrivals = np.sort(rng.uniform(0, duration, n))
    return arrivals

# -------------------------------------------------
# Dead-time models
# -------------------------------------------------
def process_nonparalyzable(arrivals, tau):
    accepted = []
    rejected = []
    dead_windows = []

    dead_until = -np.inf

    for t in arrivals:
        if t >= dead_until:
            accepted.append(t)
            dead_windows.append((t, t + tau))
            dead_until = t + tau
        else:
            rejected.append(t)

    return np.array(accepted), np.array(rejected), dead_windows


def process_paralyzable(arrivals, tau):
    accepted = []
    rejected = []
    dead_windows = []

    dead_until = -np.inf
    current_dead_start = None

    for t in arrivals:
        if t >= dead_until:
            accepted.append(t)
            current_dead_start = t
            dead_until = t + tau
            dead_windows.append([current_dead_start, dead_until])
        else:
            rejected.append(t)
            dead_until = t + tau
            dead_windows[-1][1] = dead_until

    return np.array(accepted), np.array(rejected), dead_windows

# -------------------------------------------------
# Build analog-like pulse traces
# -------------------------------------------------
def build_trace(t_axis, arrivals, accepted, width):
    incoming_trace = np.zeros_like(t_axis)
    detected_trace = np.zeros_like(t_axis)

    for t0 in arrivals:
        incoming_trace += make_pulse(t_axis, t0, width, amp=0.7)

    for t0 in accepted:
        detected_trace += make_pulse(t_axis, t0, width, amp=1.0)

    return incoming_trace, detected_trace

# -------------------------------------------------
# Theoretical curves
# -------------------------------------------------
def measured_nonparalyzable(R, tau):
    return R / (1 + R * tau)

def measured_paralyzable(R, tau):
    return R * np.exp(-R * tau)

# -------------------------------------------------
# Plotting
# -------------------------------------------------
t_axis = np.arange(0, duration, dt)

fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.2, 1])

ax_np = fig.add_subplot(gs[0])
ax_p  = fig.add_subplot(gs[1])
ax_curve = fig.add_subplot(gs[2])

plt.subplots_adjust(left=0.08, right=0.97, top=0.96, bottom=0.22, hspace=0.38)

def redraw(rate, tau, seed=None):
    global rng
    if seed is not None:
        rng = np.random.default_rng(seed)

    arrivals = generate_arrivals(rate, duration, rng)

    acc_np, rej_np, win_np = process_nonparalyzable(arrivals, tau)
    acc_p, rej_p, win_p = process_paralyzable(arrivals, tau)

    incoming_np, detected_np = build_trace(t_axis, arrivals, acc_np, pulse_width)
    incoming_p, detected_p = build_trace(t_axis, arrivals, acc_p, pulse_width)

    ax_np.clear()
    ax_p.clear()
    ax_curve.clear()

    # ---------------- Non-paralyzable panel ----------------
    for start, end in win_np:
        ax_np.axvspan(start, end, alpha=0.15)

    ax_np.plot(t_axis, incoming_np, label="Incoming signal")
    ax_np.plot(t_axis, detected_np, label="Recorded signal")

    ax_np.scatter(arrivals, np.full_like(arrivals, -0.05), s=18, marker='|', label="Arrivals")
    if len(acc_np) > 0:
        ax_np.scatter(acc_np, np.full_like(acc_np, -0.12), s=30, marker='o', label="Accepted")
    if len(rej_np) > 0:
        ax_np.scatter(rej_np, np.full_like(rej_np, -0.19), s=30, marker='x', label="Rejected")

    ax_np.set_title("Non-paralyzable detector")
    ax_np.set_ylabel("Signal")
    ax_np.set_xlim(0, duration)
    ax_np.set_ylim(-0.28, max(1.8, incoming_np.max(), detected_np.max()) + 0.1)
    ax_np.grid(alpha=0.25)
    ax_np.legend(loc="upper right")

    # ---------------- Paralyzable panel ----------------
    for start, end in win_p:
        ax_p.axvspan(start, end, alpha=0.15)

    ax_p.plot(t_axis, incoming_p, label="Incoming signal")
    ax_p.plot(t_axis, detected_p, label="Recorded signal")

    ax_p.scatter(arrivals, np.full_like(arrivals, -0.05), s=18, marker='|', label="Arrivals")
    if len(acc_p) > 0:
        ax_p.scatter(acc_p, np.full_like(acc_p, -0.12), s=30, marker='o', label="Accepted")
    if len(rej_p) > 0:
        ax_p.scatter(rej_p, np.full_like(rej_p, -0.19), s=30, marker='x', label="Rejected")

    ax_p.set_title("Paralyzable detector")
    ax_p.set_xlabel("Time (s)")
    ax_p.set_ylabel("Signal")
    ax_p.set_xlim(0, duration)
    ax_p.set_ylim(-0.28, max(1.8, incoming_p.max(), detected_p.max()) + 0.1)
    ax_p.grid(alpha=0.25)
    ax_p.legend(loc="upper right")

    # ---------------- Rate curve panel ----------------
    R = np.linspace(1, max(5000, rate * 2.2), 500)
    m_np = measured_nonparalyzable(R, tau)
    m_p = measured_paralyzable(R, tau)

    ax_curve.plot(R, m_np, label="Non-paralyzable")
    ax_curve.plot(R, m_p, label="Paralyzable")
    ax_curve.axvline(rate, linestyle='--', alpha=0.7)

    measured_np_sim = len(acc_np) / duration
    measured_p_sim = len(acc_p) / duration

    ax_curve.scatter([rate], [measured_np_sim], s=60, label="Simulated NP")
    ax_curve.scatter([rate], [measured_p_sim], s=60, marker='s', label="Simulated P")

    ax_curve.set_title("Measured count rate vs true count rate")
    ax_curve.set_xlabel("True count rate (counts/s)")
    ax_curve.set_ylabel("Measured count rate (counts/s)")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(loc="best")

    np_text = (
        f"NP: true={len(arrivals)/duration:.0f} cps, "
        f"recorded={measured_np_sim:.0f} cps, lost={len(rej_np)}"
    )
    p_text = (
        f"P: true={len(arrivals)/duration:.0f} cps, "
        f"recorded={measured_p_sim:.0f} cps, lost={len(rej_p)}"
    )

    ax_np.text(0.01, 0.92, np_text, transform=ax_np.transAxes, va='top')
    ax_p.text(0.01, 0.92, p_text, transform=ax_p.transAxes, va='top')

    fig.canvas.draw_idle()

# -------------------------------------------------
# Initial draw
# -------------------------------------------------
redraw(rate0, tau0, seed=1)

# -------------------------------------------------
# Sliders
# -------------------------------------------------
ax_rate = plt.axes([0.12, 0.11, 0.72, 0.03])
ax_tau = plt.axes([0.12, 0.06, 0.72, 0.03])

s_rate = Slider(ax_rate, "True rate (counts/s)", 50, 4000, valinit=rate0, valstep=10)
s_tau = Slider(ax_tau, "Dead time (ms)", 0.1, 5.0, valinit=tau0 * 1000, valstep=0.05)

def update(_):
    rate = s_rate.val
    tau = s_tau.val / 1000.0
    redraw(rate, tau)

s_rate.on_changed(update)
s_tau.on_changed(update)

# -------------------------------------------------
# New random realization button
# -------------------------------------------------
ax_button = plt.axes([0.86, 0.055, 0.1, 0.08])
btn = Button(ax_button, "New events")

def reroll(event):
    rate = s_rate.val
    tau = s_tau.val / 1000.0
    redraw(rate, tau, seed=np.random.randint(0, 1_000_000))

btn.on_clicked(reroll)

plt.show()