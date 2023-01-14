import numpy as np
import plotly.graph_objects as go
import plotly.subplots


class LIF:
    def __init__(self):
        self.tau_m = 8.0  # Membrane time constant [ms]
        self.abs_ref = 3.0  # Absolute refractory period [ms]
        self.R_m = 2.6  # Membrane resistance [MΩ]
        self.E_L = -70.0  # Leakage reversal potential [mV]
        self.theta = -50.0  # Spiking threshold [mV]
        self.time_from_spike = None  # Time elapsed from last spike [ms]

        self.u = self.E_L  # Membrane potential [mV]
        self.u_spike = 20.0  # Spiking decoration [mV]

    def step(self, input_current, dt):
        if self.time_from_spike is not None and self.time_from_spike < self.abs_ref:
            self.u = self.E_L
            self.time_from_spike += dt
        elif self.u == self.u_spike:
            self.u = self.E_L
        elif self.u >= self.theta:
            self.u = self.u_spike
            self.time_from_spike = 0.0
        else:
            du = (-(self.u - self.E_L) + self.R_m * input_current) / self.tau_m
            self.u += du * dt


simulation_time = 1000  # Total simulation time [ms]
dt = 0.01  # Time step [ms]
time = np.arange(0, simulation_time, dt)  # Discrete time [ms]

theta = 0.03  # Rate of mean reversion
mu = 8.0  # Long-term mean
sigma = 0.9  # Noise amplitude

input_current = np.zeros(len(time))  # Input current [nA]
eta = np.random.normal(0, 1, len(time))

for t in range(len(time) - 1):
    # Numerical simulation of Ornstein-Uhlenbeck process with Euler-Maruyama method
    input_current[t + 1] = input_current[t] + theta * (mu - input_current[t]) * dt + sigma * np.sqrt(dt) * eta[t]

neuron = LIF()
u = []

for t in range(len(time)):
    neuron.step(input_current[t], dt)
    u.append(neuron.u)

fig = plotly.subplots.make_subplots(rows=2, cols=1)

fig.add_trace(go.Scatter(x=time, y=u), row=1, col=1)
fig.add_trace(go.Scatter(x=time, y=input_current), row=2, col=1)

fig.update_xaxes(title_text="t [ms]")
fig.update_yaxes(title_text="Membrane potential [mV]", row=1, col=1)
fig.update_yaxes(title_text="Input current [nA]", row=2, col=1)

fig.update_layout(template="simple_white", showlegend=False)
fig.update_traces(line_color="#000000")

fig.show()
