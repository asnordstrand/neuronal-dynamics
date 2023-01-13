import numpy as np
import plotly.graph_objects as go
import plotly.subplots

simulation_time = 1000  # Total simulation time [ms]
dt = 0.01  # Time step [ms]
time = np.arange(0, simulation_time, dt)  # Discrete time [ms]

theta = 0.03  # Rate of mean reversion
mu = 8.0  # Long-term mean
sigma = 0.9  # Noise amplitude

x = np.zeros(len(time))  # Input current [nA]
eta = np.random.normal(0, 1, len(time))

for t in range(len(time) - 1):
    # Numerical simulation of Ornstein-Uhlenbeck process with Euler-Maruyama method
    x[t + 1] = x[t] + theta * (mu - x[t]) * dt + sigma * np.sqrt(dt) * eta[t]

tau_m = 8.0  # Membrane time constant [ms]
abs_ref = 3.0  # Absolute refractory period [ms]
R_m = 2.6  # Membrane resistance [MΩ]
E_L = -70.0  # Leakage reversal potential [mV]
u_thresh = -50.0  # Spiking threshold [mV]

u = np.zeros(len(time))  # Membrane potential [mV]
u[0] = E_L

t_hat = 0.0  # Last spike time [ms]
for t in range(len(time) - 1):
    if t - t_hat < abs_ref / dt:
        u[t] = E_L

    if u[t] >= u_thresh:
        u[t] = 20.0
        u[t + 1] = E_L
        t_hat = t
    else:
        du = (-(u[t] - E_L) + R_m * x[t]) / tau_m
        u[t + 1] = u[t] + dt * du

fig = plotly.subplots.make_subplots(rows=2, cols=1)

fig.add_trace(go.Scatter(x=time, y=u), row=1, col=1)
fig.add_trace(go.Scatter(x=time, y=x), row=2, col=1)

fig.update_xaxes(title_text="t [ms]")
fig.update_yaxes(title_text="Membrane potential [mV]", row=1, col=1)
fig.update_yaxes(title_text="Input current [nA]", row=2, col=1)

fig.update_layout(template="simple_white", showlegend=False)
fig.update_traces(line_color="#000000")

fig.show()
