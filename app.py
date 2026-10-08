"""
app.py — MomentumPlane Interactive Dashboard (streamlit run app.py)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import streamlit as st
from momentum_plane import MomentumPlanePipeline, SimulationConfig, Visualizer

st.set_page_config(page_title="MomentumPlane", page_icon="wave", layout="wide")

st.sidebar.title("Simulation Controls")
with st.sidebar.expander("Injector", expanded=True):
    grid_size = st.selectbox("Grid Size", [32, 64, 128, 256], index=1)
    stride = st.slider("Injection Stride", 2, 32, 8)
    kx = st.slider("Wavevector kx", -3.0, 3.0, 0.8, 0.1)
    ky = st.slider("Wavevector ky", -3.0, 3.0, 0.0, 0.1)
    sigma = st.slider("Packet Width", 0.5, 10.0, 3.0, 0.5)
    phase_jitter = st.slider("Phase Jitter", 0.0, 2.0, 0.0, 0.1)
with st.sidebar.expander("LatticeHop", expanded=True):
    n_steps = st.slider("Evolution Steps", 1, 100, 30)
    coin_type = st.selectbox("Coin Operator", ["hadamard", "grover"])
    coin_angle = st.slider("Coin Phase Bias", 0.0, 3.14, 0.0, 0.05)
    boundary = st.selectbox("Boundary", ["periodic", "reflective"])
with st.sidebar.expander("Display", expanded=False):
    record_every = st.slider("Record Every N Steps", 1, 10, 2)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)
run_button = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)

st.title("MomentumPlane")
st.caption("Physics-inspired simulation for periodic coherent injection and discrete quantum hopping on momentum-space lattices.")
st.markdown("**Pipeline:** Inject coherent packets -> DTQW evolution -> 2D FFT -> Watch diffraction peaks converge.")

if run_button or "result" not in st.session_state:
    with st.spinner("Evolving quantum state..."):
        cfg = SimulationConfig(
            grid_size=grid_size, stride=stride, wavevector=(kx, ky),
            sigma=sigma, phase_jitter=phase_jitter, n_steps=n_steps,
            coin_type=coin_type, coin_angle=coin_angle, boundary=boundary,
            record_every=record_every, seed=int(seed),
        )
        result = MomentumPlanePipeline(cfg).run()
        st.session_state.result = result
        st.session_state.cfg = cfg
else:
    result = st.session_state.result
    cfg = st.session_state.cfg

if "result" in st.session_state:
    result = st.session_state.result
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Position Space")
        st.caption("Probability density after evolution")
        st.pyplot(Visualizer(use_log=True).heatmap(
            result["final_position"] + 1e-12, title=f"After {cfg.n_steps} steps", xlabel="x", ylabel="y"))
    with col2:
        st.subheader("Momentum Plane")
        st.caption("Diffraction / convergence pattern")
        st.pyplot(Visualizer(use_log=True).heatmap(
            result["final_momentum_intensity"] + 1e-12, title="Momentum-space intensity", xlabel="kx", ylabel="ky"))
    st.subheader("Convergence Dynamics")
    ca, cb = st.columns(2)
    with ca:
        st.pyplot(Visualizer().convergence_plot(result["peak_intensities"]))
    with cb:
        st.pyplot(Visualizer().convergence_plot(result["n_peaks"], title="Number of Momentum Peaks"))
    st.subheader("Simulation Stats")
    sc = st.columns(4)
    sc[0].metric("Grid", f"{cfg.grid_size}x{cfg.grid_size}")
    sc[1].metric("Injection Sites", f"{(cfg.grid_size // cfg.stride) ** 2}")
    sc[2].metric("Evolution Steps", cfg.n_steps)
    sc[3].metric("Final Peaks", result["n_peaks"][-1] if result["n_peaks"] else 0)
    with st.expander("Compare: Before vs After Evolution"):
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(Visualizer(use_log=True).heatmap(
                result["initial_momentum"] + 1e-12, title="Momentum Plane - Step 0"))
        with c2:
            st.pyplot(Visualizer(use_log=True).heatmap(
                result["final_momentum_intensity"] + 1e-12, title=f"Momentum Plane - Step {cfg.n_steps}"))
