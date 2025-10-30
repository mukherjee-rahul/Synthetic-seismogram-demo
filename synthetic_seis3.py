import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dataclasses import dataclass

# ------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------
@dataclass
class Grid:
    x_start: float = 0.0
    x_end: float = 5000.0
    dx: float = 50.0
    z_max: float = 2000.0
    dz: float = 5.0

@dataclass
class TimeSampling:
    dt: float = 0.002
    t_max: float = 2.5

@dataclass
class Wavelet:
    f_dom: float = 25.0
    length: float = 0.256

@dataclass
class Layers:
    velocities: tuple = (1400.0, 2400.0, 2000.0, 2700.0, 3400.0)

# ------------------------------------------------------------
# Core Functions
# ------------------------------------------------------------
def ricker(f, length, dt):
    n = int(np.round(length / dt))
    if n % 2 == 0:
        n += 1
    t = (np.arange(n) - n // 2) * dt
    pi2f2t2 = (np.pi**2) * (f**2) * (t**2)
    w = (1.0 - 2.0 * pi2f2t2) * np.exp(-pi2f2t2)
    w /= np.max(np.abs(w))
    return w, t

# ------------------------------------------------------------
# Reflector Function Generator
# ------------------------------------------------------------
def reflector_function(x, base_depth, shape):
    """Return z(x) for a single reflector."""
    if shape == "flat":
        return base_depth + 0.0 * x

    elif shape == "gaussian":
        return base_depth - 150.0 * np.exp(-((x - 2500.0) ** 2) / (2.0 * (800.0 ** 2)))

    elif shape == "sine-cosine":
        return base_depth + 100.0 * np.sin(2 * np.pi * x / 2500.0)

    elif shape == "sigmoid":
        return base_depth + 150.0 / (1 + np.exp(-(x - 2500.0) / 400.0))

    elif shape == "sloping line":
        return base_depth + 0.15 * (x - x[0])

    else:
        return base_depth + 0.0 * x  # fallback to flat


def build_reflectors(x, shapes):
    """Build multiple reflectors with user-selected shapes."""
    base_depths = [400.0, 800.0, 1200.0, 1750.0]
    reflectors = []
    for i, shape in enumerate(shapes):
        reflectors.append(reflector_function(x, base_depths[i], shape))
    Z = np.vstack(reflectors)
    Z_sorted = np.sort(Z, axis=0)
    return Z_sorted

# ------------------------------------------------------------
# Velocity and Reflectivity Computation
# ------------------------------------------------------------
def assign_velocity_model(x, z, reflectors, velocities):
    n_ref = reflectors.shape[0]
    n_layers = n_ref + 1
    nx, nz = x.size, z.size
    v = np.zeros((nz, nx))
    for ix in range(nx):
        bounds = reflectors[:, ix]
        for iz in range(nz):
            z_here = z[iz]
            k = np.searchsorted(bounds, z_here, side="right")
            v[iz, ix] = velocities[k]
    return v


def two_way_time_curve(z, v_depth):
    v_mid = 0.5 * (v_depth[:-1] + v_depth[1:])
    dt_mid = 2.0 * (np.diff(z) / v_mid)
    t = np.zeros_like(z)
    t[1:] = np.cumsum(dt_mid)
    return t


def depth_reflectivity_from_velocity(v_depth):
    rc = np.zeros_like(v_depth)
    v1 = v_depth[:-1]
    v2 = v_depth[1:]
    denom = v2 + v1
    mask = denom != 0.0
    rc[1:][mask] = (v2[mask] - v1[mask]) / denom[mask]
    return rc


def map_depth_rc_to_time(rc_depth, t_of_z, dt, t_max):
    nt = int(np.floor(t_max / dt)) + 1
    rc_time = np.zeros(nt)
    times = t_of_z
    idx = np.clip(np.round(times / dt).astype(int), 0, nt - 1)
    np.add.at(rc_time, idx, rc_depth)
    return rc_time


def convolve_traces(rc_t, wavelet):
    nt, nx = rc_t.shape
    out = np.zeros_like(rc_t)
    for ix in range(nx):
        out[:, ix] = np.convolve(rc_t[:, ix], wavelet[::-1], mode='same')  # polarity fix
    return out


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
def wiggle_plot(seis, x, t, scale=0.5, title="Wiggle Plot"):
    fig, ax = plt.subplots(figsize=(10, 6))
    for ix, xpos in enumerate(x):
        trace = seis[:, ix]
        ax.plot(xpos + trace * scale, t, 'k', linewidth=0.5)
        ax.fill_betweenx(t, xpos, xpos + np.maximum(trace, 0) * scale, color='k', alpha=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Distance x (m)")
    ax.set_ylabel("Two-way time t (s)")
    ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig)


# ------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------
def main():
    st.title("🎧 2D Synthetic Seismogram Builder (Per-Reflector Function)")

    # Sidebar Inputs
    st.sidebar.header("Model Parameters")

    f_dom = st.sidebar.slider("Ricker Wavelet Dominant Frequency (Hz)", 5, 60, 25)
    vel_input = st.sidebar.text_input("Layer Velocities (comma separated)", "1400,2400,2000,2700,3400")
    velocities = tuple(map(float, vel_input.split(",")))

    st.sidebar.markdown("### Reflector Shapes")
    reflector_shapes = []
    for i in range(4):
        shape = st.sidebar.selectbox(
            f"Reflector {i+1} Shape",
            ["flat", "gaussian", "sine-cosine", "sigmoid", "sloping line"],
            index=0,
            key=f"shape{i}"
        )
        reflector_shapes.append(shape)

    GRID = Grid()
    TS = TimeSampling()
    WAV = Wavelet(f_dom=f_dom)
    LAY = Layers(velocities=velocities)

    x = np.arange(GRID.x_start, GRID.x_end + GRID.dx, GRID.dx)
    z = np.arange(0.0, GRID.z_max + GRID.dz, GRID.dz)
    nx, nz = x.size, z.size

    reflectors = build_reflectors(x, reflector_shapes)
    v = assign_velocity_model(x, z, reflectors, LAY.velocities)

    nt = int(np.floor(TS.t_max / TS.dt)) + 1
    rc_time = np.zeros((nt, nx))
    t_axis = np.arange(nt) * TS.dt

    for ix in range(nx):
        vcol = v[:, ix]
        t_of_z = two_way_time_curve(z, vcol)
        rc_depth = depth_reflectivity_from_velocity(vcol)
        rc_t = map_depth_rc_to_time(rc_depth, t_of_z, TS.dt, TS.t_max)
        rc_time[:, ix] = rc_t

    w, t_w = ricker(WAV.f_dom, WAV.length, TS.dt)
    seis = convolve_traces(rc_time, w)

    # Velocity Model
    st.subheader("Velocity Model")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    im1 = ax1.imshow(v, extent=[x[0], x[-1], z[-1], z[0]], aspect='auto')
    for i in range(reflectors.shape[0]):
        ax1.plot(x, reflectors[i, :], 'k', linewidth=1.0)
    plt.colorbar(im1, ax=ax1, label='Velocity (m/s)')
    st.pyplot(fig1)

    # Reflectivity Section
    st.subheader("Reflectivity Section")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    im2 = ax2.imshow(rc_time, extent=[x[0], x[-1], t_axis[-1], t_axis[0]], aspect='auto')
    plt.colorbar(im2, ax=ax2, label='Reflectivity')
    st.pyplot(fig2)

    # Synthetic Seismogram
    st.subheader("Synthetic Seismic Section")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    im3 = ax3.imshow(seis, extent=[x[0], x[-1], t_axis[-1], t_axis[0]], aspect='auto')
    plt.colorbar(im3, ax=ax3, label='Amplitude')
    st.pyplot(fig3)

    wiggle_plot(seis, x, t_axis, scale=500, title="Seismic Wiggle Plot")

    # Inverse Divergence Correction
    v_t = np.zeros_like(seis)
    for ix in range(len(x)):
        v_depth = v[:, ix]
        t_of_z = two_way_time_curve(z, v_depth)
        v_t[:, ix] = np.interp(t_axis, t_of_z, v_depth, left=v_depth[0], right=v_depth[-1])

    t_axis_safe = t_axis.copy()
    t_axis_safe[0] = np.nan
    correction = v_t * t_axis_safe[:, np.newaxis]
    seis_invdiv = seis / correction

    st.subheader("Inverse Divergence Corrected Section")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    im4 = ax4.imshow(seis_invdiv, extent=[x[0], x[-1], t_axis[-1], t_axis[0]], aspect='auto')
    plt.colorbar(im4, ax=ax4, label='Amplitude/(v·t)')
    st.pyplot(fig4)

    wiggle_plot(seis_invdiv, x, t_axis, scale=800000, title="Inverse Divergence Corrected Wiggle Plot")


if __name__ == "__main__":
    main()
