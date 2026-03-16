import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Detector Dead Time Simulator")
st.write("Interactive comparison of **Paralyzable** and **Non-Paralyzable** detector models.")

# ---------------------------------------------
# User Controls
# ---------------------------------------------
rate = st.slider("True count rate (counts/s)", 50, 4000, 800)
tau_ms = st.slider("Dead time (ms)", 0.1, 5.0, 1.5)
tau = tau_ms / 1000

duration = 0.02
dt = 1e-5
pulse_width = 0.00015

rng = np.random.default_rng()

# ---------------------------------------------
# Pulse generator
# ---------------------------------------------
def make_pulse(t_axis, t0, width, amp=1.0):
    sigma = width / 5
    return amp * np.exp(-0.5 * ((t_axis - t0) / sigma) ** 2)

# ---------------------------------------------
# Generate arrivals
# ---------------------------------------------
def generate_arrivals(rate, duration):
    n = rng.poisson(rate * duration)
    return np.sort(rng.uniform(0, duration, n))

# ---------------------------------------------
# Dead time models
# ---------------------------------------------
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

    for t in arrivals:

        if t >= dead_until:

            accepted.append(t)
            dead_until = t + tau
            dead_windows.append([t, dead_until])

        else:

            rejected.append(t)
            dead_until = t + tau
            dead_windows[-1][1] = dead_until

    return np.array(accepted), np.array(rejected), dead_windows

# ---------------------------------------------
# Rate theory
# ---------------------------------------------
def measured_nonparalyzable(R, tau):
    return R / (1 + R * tau)

def measured_paralyzable(R, tau):
    return R * np.exp(-R * tau)

# ---------------------------------------------
# Simulation
# ---------------------------------------------
t_axis = np.arange(0, duration, dt)

arrivals = generate_arrivals(rate, duration)

acc_np, rej_np, win_np = process_nonparalyzable(arrivals, tau)
acc_p, rej_p, win_p = process_paralyzable(arrivals, tau)

# ---------------------------------------------
# Pulse traces
# ---------------------------------------------
def build_trace(arrivals, accepted):

    incoming = np.zeros_like(t_axis)
    detected = np.zeros_like(t_axis)

    for t in arrivals:
        incoming += make_pulse(t_axis, t, pulse_width, 0.7)

    for t in accepted:
        detected += make_pulse(t_axis, t, pulse_width, 1.0)

    return incoming, detected


incoming_np, detected_np = build_trace(arrivals, acc_np)
incoming_p, detected_p = build_trace(arrivals, acc_p)

# ---------------------------------------------
# Plot
# ---------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(10, 10),
    gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.45}
)

# Non-paralyzable
for s,e in win_np:
    ax1.axvspan(s,e,alpha=0.15)

ax1.plot(t_axis,incoming_np,label="Incoming")
ax1.plot(t_axis,detected_np,label="Detected")

ax1.scatter(arrivals,np.full_like(arrivals,-0.05),marker="|")
ax1.scatter(acc_np,np.full_like(acc_np,-0.1),marker="o")
ax1.scatter(rej_np,np.full_like(rej_np,-0.15),marker="x")

ax1.set_title("Non-Paralyzable Detector")
ax1.set_xlim(0,duration)
ax1.grid(alpha=0.3)
ax1.legend()

# Paralyzable
for s,e in win_p:
    ax2.axvspan(s,e,alpha=0.15)

ax2.plot(t_axis,incoming_p,label="Incoming")
ax2.plot(t_axis,detected_p,label="Detected")

ax2.scatter(arrivals,np.full_like(arrivals,-0.05),marker="|")
ax2.scatter(acc_p,np.full_like(acc_p,-0.1),marker="o")
ax2.scatter(rej_p,np.full_like(rej_p,-0.15),marker="x")

ax2.set_title("Paralyzable Detector")
ax2.set_xlim(0,duration)
ax2.grid(alpha=0.3)
ax2.legend()

# Count rate curve
R = np.linspace(1,5000,500)

ax3.plot(R,measured_nonparalyzable(R,tau),label="Non-paralyzable")
ax3.plot(R,measured_paralyzable(R,tau),label="Paralyzable")

ax3.axvline(rate,linestyle="--")

ax3.scatter(rate,len(acc_np)/duration,label="Sim NP")
ax3.scatter(rate,len(acc_p)/duration,marker="s",label="Sim P")

ax3.set_xlabel("True count rate")
ax3.set_ylabel("Measured rate")
ax3.set_title("Measured vs True Count Rate")
ax3.grid(alpha=0.3)
ax3.legend()

st.pyplot(fig)

# ---------------------------------------------
# Statistics
# ---------------------------------------------
st.subheader("Simulation Statistics")

col1,col2 = st.columns(2)

with col1:
    st.write("Non-Paralyzable")
    st.write("Recorded rate:", round(len(acc_np)/duration))
    st.write("Lost events:", len(rej_np))

with col2:
    st.write("Paralyzable")
    st.write("Recorded rate:", round(len(acc_p)/duration))
    st.write("Lost events:", len(rej_p))

if st.button("Generate New Random Events"):
    st.experimental_rerun()
