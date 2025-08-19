# pmsm_pwm_sim.py
# Self-contained PMSM + PWM speed control simulator (surface-mounted PMSM)
# Run: python pmsm_pwm_sim.py

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class PMSMParams:
    p: int = 4
    Rs: float = 0.3
    Ld: float = 1.6e-3
    Lq: float = 1.6e-3
    psi_f: float = 0.06
    J: float = 2.0e-4
    B: float = 1.0e-4
    Vdc: float = 48.0

@dataclass
class CtrlGains:
    Kp_i: float = 25.0
    Ki_i: float = 6000.0
    Kp_w: float = 0.01
    Ki_w: float = 1.5

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def inv_park(vd, vq, theta_e):
    c, s = np.cos(theta_e), np.sin(theta_e)
    v_alpha =  c * vd - s * vq
    v_beta  =  s * vd + c * vq
    return v_alpha, v_beta

def inv_clarke(v_alpha, v_beta):
    va = v_alpha
    vb = -0.5*v_alpha + (np.sqrt(3)/2)*v_beta
    vc = -0.5*v_alpha - (np.sqrt(3)/2)*v_beta
    return va, vb, vc

def park(i_alpha, i_beta, theta_e):
    c, s = np.cos(theta_e), np.sin(theta_e)
    id_ =  c * i_alpha + s * i_beta
    iq_ = -s * i_alpha + c * i_beta
    return id_, iq_

def clarke(ia, ib, ic):
    i_alpha = ia
    i_beta  = (ia + 2*ib)/np.sqrt(3)
    return i_alpha, i_beta

def main():
    params = PMSMParams()
    gains = CtrlGains()
    dt = 1e-5
    t_end = 0.3
    N = int(t_end/dt)

    def wref_profile(t):
        if t < 0.05:
            return 0.0
        elif t < 0.2:
            return 1000.0 * 2*np.pi/60.0
        else:
            return 1500.0 * 2*np.pi/60.0

    def load_profile(t):
        return 0.05 if t > 0.12 else 0.0

    ia = ib = ic = 0.0
    id_ = iq_ = 0.0
    w_m = theta_m = theta_e = 0.0
    int_id = int_iq = int_w = 0.0

    t_log = np.zeros(N)
    w_m_log = np.zeros(N)
    w_ref_log = np.zeros(N)
    id_log = np.zeros(N)
    iq_log = np.zeros(N)
    vd_log = np.zeros(N)
    vq_log = np.zeros(N)
    Te_log = np.zeros(N)

    for k in range(N):
        t = k*dt
        w_ref = wref_profile(t)
        Tl = load_profile(t)

        i_alpha, i_beta = clarke(ia, ib, ic)
        id_, iq_ = park(i_alpha, i_beta, theta_e)

        e_w = w_ref - w_m
        int_w += e_w * dt
        iq_ref = gains.Kp_w * e_w + gains.Ki_w * int_w
        iq_ref = clamp(iq_ref, -60.0, 60.0)
        id_ref = 0.0

        e_id = id_ref - id_
        e_iq = iq_ref - iq_
        int_id += e_id * dt
        int_iq += e_iq * dt

        we = params.p * w_m
        vd_ff =  params.Rs*id_ - params.Lq * iq_ * we
        vq_ff =  params.Rs*iq_ + (params.Ld * id_ + params.psi_f) * we

        vd_star = gains.Kp_i * e_id + gains.Ki_i * int_id + vd_ff
        vq_star = gains.Kp_i * e_iq + gains.Ki_i * int_iq + vq_ff

        v_alpha, v_beta = inv_park(vd_star, vq_star, theta_e)
        va_star, vb_star, vc_star = inv_clarke(v_alpha, v_beta)

        v_max = params.Vdc * 0.5
        va = np.clip(va_star, -v_max, v_max)
        vb = np.clip(vb_star, -v_max, v_max)
        vc = np.clip(vc_star, -v_max, v_max)

        did_dt = (vd_star - params.Rs*id_ + we*params.Lq*iq_) / params.Ld
        diq_dt = (vq_star - params.Rs*iq_ - we*(params.Ld*id_ + params.psi_f)) / params.Lq
        id_ += did_dt * dt
        iq_ += diq_dt * dt

        Te = 1.5 * params.p * (params.psi_f * iq_)
        dwm_dt = (Te - Tl - params.B*w_m) / params.J
        w_m += dwm_dt * dt
        theta_m = (theta_m + w_m * dt) % (2*np.pi)
        theta_e = (theta_e + we * dt) % (2*np.pi)

        t_log[k] = t
        w_m_log[k] = w_m
        w_ref_log[k] = w_ref
        id_log[k] = id_
        iq_log[k] = iq_
        vd_log[k] = vd_star
        vq_log[k] = vq_star
        Te_log[k] = Te

        # reconstruct abc currents for next step (zero-seq=0)
        i_alpha =  np.cos(theta_e)*id_ - np.sin(theta_e)*iq_
        i_beta  =  np.sin(theta_e)*id_ + np.cos(theta_e)*iq_
        ia = i_alpha
        ib = (-0.5*i_alpha + (np.sqrt(3)/2)*i_beta)
        ic = (-0.5*i_alpha - (np.sqrt(3)/2)*i_beta)

    plt.figure(figsize=(8,4))
    plt.plot(t_log, w_m_log, label="ω_m")
    plt.plot(t_log, w_ref_log, linestyle="--", label="ω_ref")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [rad/s]")
    plt.title("Mechanical speed")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(t_log, iq_log, label="i_q")
    plt.plot(t_log, id_log, label="i_d")
    plt.xlabel("Time [s]")
    plt.ylabel("Current [A]")
    plt.title("dq currents")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(t_log, vd_log, label="v_d*")
    plt.plot(t_log, vq_log, label="v_q*")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title("dq voltage refs (pre-clamp)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(t_log, Te_log)
    plt.xlabel("Time [s]")
    plt.ylabel("Torque [N·m]")
    plt.title("Electromagnetic torque")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
