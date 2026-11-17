# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.stats import norm

#==============================================================================
# Model Inputs
#==============================================================================

PD = 0.05
LGD = 0.45
EAD = 100
rho = 0.30

n_simulations = 100000

#==============================================================================
# Generate Uniform Variables
#==============================================================================

rng = np.random.default_rng()

U1 = rng.uniform(0,1,n_simulations)
U2 = rng.uniform(0,1,n_simulations)

#==============================================================================
# Transform to Standard Normal
#==============================================================================

Z1 = norm.ppf(U1)
Z2 = norm.ppf(U2)

#==============================================================================
# Introduce Correlation
#==============================================================================

Z1_star = Z1

Z2_star = (
    rho*Z1
    +
    np.sqrt(1-rho**2)*Z2)

#==============================================================================
# Transform back to Uniform
#==============================================================================

U1_star = norm.cdf(Z1_star)
U2_star = norm.cdf(Z2_star)

#==============================================================================
# Generate default indicators
#==============================================================================

D1 = (U1_star < PD).astype(int)
D2 = (U2_star < PD).astype(int)

#==============================================================================
# Calculate loan losses
#==============================================================================

L1 = D1*LGD*EAD
L2 = D2*LGD*EAD

portfolio_loss = L1+L2

#==============================================================================
# Summarize simulation results
#==============================================================================

print("Simulation Results")
print("-----------------------------")

print("Loan 1 default rate:", D1.mean())
print("Loan 2 default rate:", D2.mean())

print("Latent correlation:",
      np.corrcoef(Z1_star, Z2_star)[0, 1])

print("Default correlation:",
      np.corrcoef(D1, D2)[0, 1])

print("Expected portfolio loss:",
      portfolio_loss.mean())

print("Maximum portfolio loss:",
      portfolio_loss.max())

# Frequency distribution of losses

loss_values, counts = np.unique(
    portfolio_loss,
    return_counts=True
)

probabilities = counts / n_simulations

for loss, probability in zip(
    loss_values, probabilities
):
    print(
        f"Loss = {loss:.2f}, "
        f"Probability = {probability:.4%}"
    )

#==============================================================================
# Plot portfolio loss distribution
#==============================================================================

import plotly.graph_objects as go

possible_losses = np.array([0, LGD*EAD, 2*LGD*EAD])

counts = np.array([
    np.sum(portfolio_loss == loss)
    for loss in possible_losses
    ])

probabilities = counts/n_simulations

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x = [f"{loss:.2f}" for loss in possible_losses],
        y = probabilities*100,
        text = [f"{p:.2%}" for p in probabilities],
        textposition = "outside",
        marker_color = ["#4C78A8", "#F2A541", "#D95F5F"],
        hovertemplate = (
            "Portfolio Loss: %{x}<br>"
            "Probability: %{y:.3f}%"
            "<extra></extra>"
            )
        )
    )

fig.update_layout(
    title="Two-Loan Portfolio Loss Distribution",
    xaxis_title="Portfolio Loss",
    yaxis_title="Probability (%)",
    template="plotly_white",
    height=500,
    bargap=0.4,
    yaxis=dict(range=[0, max(probabilities * 100) * 1.15])
)

fig.show(renderer="browser")


import plotly.express as px

fig_scatter = px.scatter(
    x=Z1_star,
    y=Z2_star,
    labels={
        "x": "Z1*",
        "y": "Z2*"
    },
    title=f"Correlated Standard Normal Variables (ρ = {rho:.2f})",
    opacity=0.3
)

fig_scatter.update_traces(
    marker=dict(size=3)
)

fig_scatter.update_layout(
    template="plotly_white",
    height=650,
    width=850
)

fig_scatter.show(renderer="browser")






