from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SimulationConfig:
    nx: int = 128
    nt: int = 600
    num_snapshots: int = 5
    dx: float = 1.0
    cfl: float = 0.45
    mu: float = 1.0
    eps_min: float = 1.2
    eps_max: float = 4.0
    src_x_min_frac: float = 0.125
    src_x_max_frac: float = 0.333
    t0_min_steps: float = 25.0
    t0_max_steps: float = 70.0
    spread_min_steps: float = 8.0
    spread_max_steps: float = 20.0
    src_freq_min: float = 0.01
    src_freq_max: float = 0.05


def build_piecewise_eps(config: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    nx = config.nx
    eps = np.ones(nx, dtype=np.float32)
    num_layers = int(rng.integers(1, 5))
    boundaries = sorted(rng.choice(np.arange(8, nx - 8), size=num_layers * 2, replace=False))

    for i in range(0, len(boundaries), 2):
        start = boundaries[i]
        end = boundaries[i + 1]
        eps[start:end] = float(rng.uniform(config.eps_min, config.eps_max))

    return eps


def gaussian_source(t: float, t0: float, spread: float, freq: float) -> float:
    envelope = np.exp(-((t - t0) ** 2) / (2.0 * spread * spread))
    carrier = np.cos(2.0 * np.pi * freq * (t - t0))
    return float(envelope * carrier)


def simulate_sample(config: SimulationConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    nx = config.nx
    nt = config.nt
    dx = config.dx
    eps = build_piecewise_eps(config, rng)

    c_max = np.sqrt(1.0 / (config.mu * np.min(eps)))
    dt = config.cfl * dx / c_max

    e = np.zeros(nx, dtype=np.float32)
    h = np.zeros(nx, dtype=np.float32)

    src_x = int(
        rng.integers(
            max(1, int(nx * config.src_x_min_frac)),
            min(nx - 1, int(nx * config.src_x_max_frac)),
        )
    )
    t0 = float(rng.uniform(config.t0_min_steps, config.t0_max_steps) * dt)
    spread = float(rng.uniform(config.spread_min_steps, config.spread_max_steps) * dt)
    freq = float(rng.uniform(config.src_freq_min, config.src_freq_max) / dt)

    snap_steps = np.linspace(0, nt - 1, config.num_snapshots, dtype=int)
    snapshots = []

    e_left_old = 0.0
    e_right_old = 0.0

    for n in range(nt):
        h[:-1] = h[:-1] + (dt / (config.mu * dx)) * (e[1:] - e[:-1])

        e[1:] = e[1:] + (dt / (eps[1:] * dx)) * (h[1:] - h[:-1])

        t = n * dt
        e[src_x] += gaussian_source(t, t0=t0, spread=spread, freq=freq)

        e[0] = e_left_old
        e_left_old = e[1]
        e[-1] = e_right_old
        e_right_old = e[-2]

        if n in snap_steps:
            snapshots.append(e.copy())

    return np.stack(snapshots, axis=0), eps


def generate_dataset(num_samples: int, config: SimulationConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    trajectories = []
    eps_profiles = []
    for _ in range(num_samples):
        traj, eps = simulate_sample(config, rng)
        trajectories.append(traj)
        eps_profiles.append(eps)
    return (
        np.stack(trajectories, axis=0).astype(np.float32),
        np.stack(eps_profiles, axis=0).astype(np.float32),
    )


def save_splits(output_dir: Path, config: SimulationConfig, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": 800,
        "val": 120,
        "test": 120,
        "test_unknown": 120,
    }

    data_train, eps_train = generate_dataset(splits["train"], config, seed=seed)
    data_val, eps_val = generate_dataset(splits["val"], config, seed=seed + 1)
    data_test, eps_test = generate_dataset(splits["test"], config, seed=seed + 2)

    unknown_config = SimulationConfig(
        nx=config.nx,
        nt=config.nt,
        num_snapshots=config.num_snapshots,
        dx=config.dx,
        cfl=config.cfl,
        mu=config.mu,
        eps_min=3.0,
        eps_max=8.0,
        src_x_min_frac=0.5,
        src_x_max_frac=0.9,
        t0_min_steps=40.0,
        t0_max_steps=90.0,
        spread_min_steps=12.0,
        spread_max_steps=28.0,
        src_freq_min=0.03,
        src_freq_max=0.08,
    )
    data_unknown, eps_unknown = generate_dataset(splits["test_unknown"], unknown_config, seed=seed + 7)

    np.save(output_dir / f"data_train_{config.nx}.npy", data_train)
    np.save(output_dir / f"data_val_{config.nx}.npy", data_val)
    np.save(output_dir / f"data_test_{config.nx}.npy", data_test)
    np.save(output_dir / f"data_test_unknown_{config.nx}.npy", data_unknown)
    np.save(output_dir / f"eps_train_{config.nx}.npy", eps_train)
    np.save(output_dir / f"eps_val_{config.nx}.npy", eps_val)
    np.save(output_dir / f"eps_test_{config.nx}.npy", eps_test)
    np.save(output_dir / f"eps_test_unknown_{config.nx}.npy", eps_unknown)

    for nx_small in [32, 64, 96]:
        small_cfg = SimulationConfig(
            nx=nx_small,
            nt=config.nt,
            num_snapshots=config.num_snapshots,
            dx=config.dx,
            cfl=config.cfl,
            mu=config.mu,
            eps_min=config.eps_min,
            eps_max=config.eps_max,
            src_x_min_frac=config.src_x_min_frac,
            src_x_max_frac=config.src_x_max_frac,
            t0_min_steps=config.t0_min_steps,
            t0_max_steps=config.t0_max_steps,
            spread_min_steps=config.spread_min_steps,
            spread_max_steps=config.spread_max_steps,
            src_freq_min=config.src_freq_min,
            src_freq_max=config.src_freq_max,
        )
        data_small_train, eps_small_train = generate_dataset(splits["train"], small_cfg, seed=seed + 100 + nx_small)
        data_small_val, eps_small_val = generate_dataset(splits["val"], small_cfg, seed=seed + 200 + nx_small)
        data_small_test, eps_small_test = generate_dataset(splits["test"], small_cfg, seed=seed + 300 + nx_small)
        np.save(output_dir / f"data_train_{nx_small}.npy", data_small_train)
        np.save(output_dir / f"data_val_{nx_small}.npy", data_small_val)
        np.save(output_dir / f"data_test_{nx_small}.npy", data_small_test)
        np.save(output_dir / f"eps_train_{nx_small}.npy", eps_small_train)
        np.save(output_dir / f"eps_val_{nx_small}.npy", eps_small_val)
        np.save(output_dir / f"eps_test_{nx_small}.npy", eps_small_test)


def build_config_from_args(args: argparse.Namespace) -> SimulationConfig:
    cfg = SimulationConfig(
        nx=args.nx,
        nt=args.nt,
        num_snapshots=args.snapshots,
        cfl=args.cfl,
    )

    if args.easy:
        cfg = SimulationConfig(
            nx=args.nx,
            nt=args.nt,
            num_snapshots=args.snapshots,
            cfl=args.cfl,
            eps_min=1.0,
            eps_max=2.2,
            src_x_min_frac=0.2,
            src_x_max_frac=0.3,
            t0_min_steps=20.0,
            t0_max_steps=40.0,
            spread_min_steps=14.0,
            spread_max_steps=26.0,
            src_freq_min=0.008,
            src_freq_max=0.02,
        )

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EM wave datasets for FNO portfolio project.")
    parser.add_argument("--output-dir", type=Path, default=Path("Data_EM"), help="Directory for .npy files")
    parser.add_argument("--nx", type=int, default=128, help="Spatial resolution")
    parser.add_argument("--nt", type=int, default=600, help="Time integration steps")
    parser.add_argument("--snapshots", type=int, default=5, help="Number of saved snapshots")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--cfl", type=float, default=0.45, help="CFL factor (smaller gives smaller dt)")
    parser.add_argument("--easy", action="store_true", help="Generate an easier, lower-complexity dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)
    save_splits(args.output_dir, cfg, seed=args.seed)
    print(f"Saved EM datasets to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
