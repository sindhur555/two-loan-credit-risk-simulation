# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:27:06 2026

@author: hp
"""


import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Two-Loan Credit Risk Simulation",
    page_icon="📊",
    layout="wide"
)

# Main heading
st.title("Two-Loan Credit Risk Simulation")

st.write(
    "An interactive Monte Carlo simulation "
    "of correlated defaults using a Gaussian copula."
)

# Sidebar inputs
st.sidebar.header("Model Parameters")

PD = st.sidebar.slider(
    "Probability of Default (%)",
    min_value=0.1,
    max_value=30.0,
    value=5.0,
    step=0.1
) / 100

LGD = st.sidebar.slider(
    "Loss Given Default (%)",
    min_value=0,
    max_value=100,
    value=45,
    step=1
) / 100

EAD = st.sidebar.number_input(
    "Exposure at Default",
    min_value=1.0,
    value=100.0,
    step=10.0
)

rho = st.sidebar.slider(
    "Latent Asset Correlation",
    min_value=0.0,
    max_value=0.99,
    value=0.30,
    step=0.01
)

n_simulations = 100000

st.sidebar.write(
    f"Number of simulations: {n_simulations:,}"
)

# Run button
run_simulation = st.sidebar.button(
    "Run Simulation",
    type="primary"
)

# Display selected parameters
st.subheader("Selected Model Parameters")

col1, col2, col3, col4 = st.columns(4)

col1.metric("PD", f"{PD:.2%}")
col2.metric("LGD", f"{LGD:.2%}")
col3.metric("EAD", f"{EAD:,.2f}")
col4.metric("Asset Correlation", f"{rho:.2f}")



# ============================================
# MONTE CARLO SIMULATION
# ============================================

import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

if run_simulation:

    # Generate independent uniform variables
    rng = np.random.default_rng()

    U1 = rng.uniform(size=n_simulations)
    U2 = rng.uniform(size=n_simulations)

    # Transform to standard normal
    Z1 = norm.ppf(U1)
    Z2 = norm.ppf(U2)

    # Generate correlated latent variables
    Z1_star = Z1
    Z2_star = (
        rho * Z1
        + np.sqrt(1 - rho**2) * Z2
    )

    # Transform back to uniform
    U1_star = norm.cdf(Z1_star)
    U2_star = norm.cdf(Z2_star)

    # Generate default indicators
    D1 = (U1_star < PD).astype(int)
    D2 = (U2_star < PD).astype(int)

    # Calculate losses
    L1 = D1 * LGD * EAD
    L2 = D2 * LGD * EAD

    portfolio_loss = L1 + L2

    # Summary statistics
    st.subheader("Simulation Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Loan 1 Default Rate",
        f"{D1.mean():.2%}"
    )

    col2.metric(
        "Loan 2 Default Rate",
        f"{D2.mean():.2%}"
    )

    col3.metric(
        "Expected Portfolio Loss",
        f"{portfolio_loss.mean():.2f}"
    )

    # Portfolio loss distribution
    possible_losses = np.array([
        0,
        LGD * EAD,
        2 * LGD * EAD
    ])

    probabilities = np.array([
        np.mean(portfolio_loss == loss)
        for loss in possible_losses
    ])

    fig_loss = go.Figure()

    fig_loss.add_trace(
        go.Bar(
            x=["No Defaults", "One Default", "Both Default"],
            y=probabilities * 100,
            text=[
                f"{p:.2%}"
                for p in probabilities
            ],
            textposition="outside"
        )
    )

    fig_loss.update_layout(
        title="Portfolio Loss Distribution",
        xaxis_title="Default Outcome",
        yaxis_title="Probability (%)",
        template="plotly_white"
    )

    st.plotly_chart(fig_loss, use_container_width=True)

    # Scatter plot of correlated latent variables
    fig_scatter = go.Figure()

    fig_scatter.add_trace(
        go.Scattergl(
            x=Z1_star,
            y=Z2_star,
            mode="markers",
            marker=dict(
                size=3,
                opacity=0.3
            )
        )
    )

    fig_scatter.update_layout(
        title="Correlated Latent Asset Variables",
        xaxis_title="Z1*",
        yaxis_title="Z2*",
        template="plotly_white",
        height=600
    )

    st.plotly_chart(fig_scatter, use_container_width=True)
