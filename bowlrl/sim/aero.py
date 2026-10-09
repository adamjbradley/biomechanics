"""Cricket-ball flight model: drag + Magnus + single pitch bounce.

Integrated with fixed-step RK4 from the release state to (a) the first ground
contact (the "landing" / pitching point) and (b) the batter's popping crease.

The aerodynamic constants are configurable; the defaults are mid-range values from
the cricket aerodynamics literature (Mehta 1985, 2005; Sayers & Hill 1999):
    C_D  ~ 0.40-0.50 for a new ball below the drag crisis (~30 m/s seam-up)
    C_L  Magnus lift from back/side spin, saturating with spin ratio S = r*omega/|v|

Swing from seam orientation is NOT modelled here (it needs a seam-angle state and an
asymmetric-boundary-layer model); the hook `extra_force` lets you add it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

BALL_MASS = 0.156          # kg (Laws of Cricket: 155.9-163 g)
BALL_RADIUS = 0.036        # m (circumference 22.4-22.9 cm)
AIR_DENSITY = 1.20         # kg/m^3 (sea level, 20 C)
G = 9.81


@dataclass
class AeroConfig:
    cd: float = 0.45
    cl_max: float = 0.28           # saturated Magnus lift coefficient
    cl_spin_scale: float = 0.12    # spin ratio at which C_L reaches half of cl_max
    restitution: float = 0.35      # vertical COR of a firm pitch (0.3 soft - 0.5 hard)
    bounce_friction: float = 0.25  # horizontal speed retained = 1 - this * tan-like factor
    spin_decay: float = 0.03       # fraction of spin lost per second in flight
    dt: float = 0.0005
    batter_crease_x: float = 17.68
    stumps_x: float = 18.90
    max_time: float = 2.0
    mass: float = BALL_MASS
    radius: float = BALL_RADIUS
    rho: float = AIR_DENSITY


@dataclass
class FlightResult:
    landed: bool
    landing_pos: np.ndarray            # (3,) first ground contact, z = radius
    landing_time: float
    landing_speed: float               # |v| just before bounce
    arrival_pos: Optional[np.ndarray]  # position when x crosses batter's popping crease
    arrival_speed: Optional[float]
    stumps_height: Optional[float]     # height when x crosses the stumps line (None if not reached)
    trajectory: np.ndarray             # (N, 7): t, x, y, z, vx, vy, vz
    full_toss: bool                    # reached the batter's crease without bouncing


def magnus_lift_coefficient(spin_ratio: float, cfg: AeroConfig) -> float:
    s = max(spin_ratio, 0.0)
    return cfg.cl_max * s / (cfg.cl_spin_scale + s)


def acceleration(v: np.ndarray, omega: np.ndarray, cfg: AeroConfig,
                 extra_force: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None) -> np.ndarray:
    speed = float(np.linalg.norm(v))
    area = np.pi * cfg.radius ** 2
    a = np.array([0.0, 0.0, -G])
    if speed < 1e-6:
        return a
    q = 0.5 * cfg.rho * area * speed ** 2
    drag = -cfg.cd * q * v / speed
    omega_norm = float(np.linalg.norm(omega))
    lift = np.zeros(3)
    if omega_norm > 1e-6:
        spin_ratio = cfg.radius * omega_norm / speed
        cl = magnus_lift_coefficient(spin_ratio, cfg)
        lift_dir = np.cross(omega / omega_norm, v / speed)
        lift = cl * q * lift_dir
    f = drag + lift
    if extra_force is not None:
        f = f + extra_force(v, omega)
    return a + f / cfg.mass


def simulate_flight(pos: np.ndarray, vel: np.ndarray, omega: np.ndarray | None = None,
                    cfg: AeroConfig | None = None,
                    extra_force: Optional[Callable] = None) -> FlightResult:
    """Integrate from the release state until the ball passes the stumps line or times out."""
    cfg = cfg or AeroConfig()
    p = np.asarray(pos, dtype=float).copy()
    v = np.asarray(vel, dtype=float).copy()
    w = np.zeros(3) if omega is None else np.asarray(omega, dtype=float).copy()
    t = 0.0
    traj = [np.concatenate([[t], p, v])]

    landed = False
    landing_pos = p.copy()
    landing_time = 0.0
    landing_speed = float(np.linalg.norm(v))
    arrival_pos = None
    arrival_speed = None
    stumps_height = None
    full_toss = False
    dt = cfg.dt

    def f(state):
        pp, vv = state[:3], state[3:]
        return np.concatenate([vv, acceleration(vv, w, cfg, extra_force)])

    state = np.concatenate([p, v])
    while t < cfg.max_time:
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        new_state = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        w *= (1.0 - cfg.spin_decay * dt)

        # ground contact
        if not landed and new_state[2] <= cfg.radius and new_state[5] < 0:
            # linear interpolation to the exact contact
            frac = (state[2] - cfg.radius) / max(state[2] - new_state[2], 1e-9)
            contact = state + frac * (new_state - state)
            landed = True
            landing_pos = contact[:3].copy()
            landing_pos[2] = cfg.radius
            landing_time = t - dt + frac * dt
            landing_speed = float(np.linalg.norm(contact[3:]))
            # bounce
            vz = -cfg.restitution * contact[5]
            horiz = contact[3:5] * (1.0 - cfg.bounce_friction * min(1.0, abs(contact[5]) / max(np.linalg.norm(contact[3:5]), 1e-6)))
            new_state = np.concatenate([landing_pos, horiz, [vz]])
            # back-spin converts partly to top-spin on bounce; crude: damp spin
            w *= 0.6

        # batter's crease crossing
        if arrival_pos is None and state[0] < cfg.batter_crease_x <= new_state[0]:
            frac = (cfg.batter_crease_x - state[0]) / max(new_state[0] - state[0], 1e-9)
            arr = state + frac * (new_state - state)
            arrival_pos = arr[:3].copy()
            arrival_speed = float(np.linalg.norm(arr[3:]))
            if not landed:
                full_toss = True
        if stumps_height is None and state[0] < cfg.stumps_x <= new_state[0]:
            frac = (cfg.stumps_x - state[0]) / max(new_state[0] - state[0], 1e-9)
            stumps_height = float(state[2] + frac * (new_state[2] - state[2]))
            state = new_state
            traj.append(np.concatenate([[t], state]))
            break
        state = new_state
        traj.append(np.concatenate([[t], state]))
        if landed and state[2] <= cfg.radius and abs(state[5]) < 0.05:
            break  # rolling

    return FlightResult(landed=landed, landing_pos=landing_pos, landing_time=landing_time,
                        landing_speed=landing_speed, arrival_pos=arrival_pos,
                        arrival_speed=arrival_speed, stumps_height=stumps_height,
                        trajectory=np.array(traj), full_toss=full_toss)


def landing_error(result: FlightResult, target_xy: np.ndarray) -> float:
    return float(np.linalg.norm(result.landing_pos[:2] - np.asarray(target_xy, dtype=float)))
