"""Record a rollout video from an untrained MSE policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from hw1_imitation.data import Normalizer, load_pusht_zarr
from hw1_imitation.model import build_policy


DEFAULT_ZARR_PATHS = (
    Path("data/pusht/pusht_cchi_v7_replay.zarr"),
    Path("/home/kris/Downloads/pusht(1)/pusht/pusht_cchi_v7_replay.zarr"),
)
ENV_ID = "gym_pusht/PushT-v0"

# Edit this one line to set the initial state:
# [agent_x, agent_y, block_x, block_y, block_angle_radians].
# Set it to None to use the environment's random reset.
CUSTOM_RESET_STATE: list[float] | None = [90.0, 420.0, 380.0, 160.0, np.deg2rad(-120.0)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out a randomly initialized MSE policy and save video."
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=None,
        help="Path to pusht_cchi_v7_replay.zarr. Defaults to a known local path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("untrained_mse_policy.mp4"),
        help="Where to save the rollout video.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--reset-to-state",
        type=float,
        nargs=5,
        metavar=("AGENT_X", "AGENT_Y", "BLOCK_X", "BLOCK_Y", "BLOCK_ANGLE"),
        default=None,
        help=(
            "Initial state [agent_x agent_y block_x block_y block_angle]. "
            "The angle is in radians."
        ),
    )
    parser.add_argument("--agent-x", type=float, default=None)
    parser.add_argument("--agent-y", type=float, default=None)
    parser.add_argument("--block-x", type=float, default=None)
    parser.add_argument("--block-y", type=float, default=None)
    parser.add_argument(
        "--block-angle",
        type=float,
        default=None,
        help="Initial block angle in radians.",
    )
    parser.add_argument(
        "--block-angle-deg",
        type=float,
        default=None,
        help="Initial block angle in degrees. Overrides --block-angle.",
    )
    return parser.parse_args()


def resolve_zarr_path(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    for candidate in DEFAULT_ZARR_PATHS:
        if candidate.exists():
            return candidate

    candidates = "\n".join(f"  - {p}" for p in DEFAULT_ZARR_PATHS)
    raise FileNotFoundError(
        "Could not find the Push-T dataset. Pass --zarr-path explicitly.\n"
        f"Checked:\n{candidates}"
    )


def resolve_reset_state(args: argparse.Namespace) -> list[float] | None:
    if CUSTOM_RESET_STATE is not None:
        return CUSTOM_RESET_STATE

    if args.reset_to_state is not None:
        return list(args.reset_to_state)

    fields = [args.agent_x, args.agent_y, args.block_x, args.block_y]
    angle = args.block_angle
    if args.block_angle_deg is not None:
        angle = np.deg2rad(args.block_angle_deg)

    if all(value is None for value in fields) and angle is None:
        return None
    if any(value is None for value in fields) or angle is None:
        raise ValueError(
            "To customize the initial state, either pass --reset-to-state with "
            "5 values, or pass all of --agent-x --agent-y --block-x --block-y "
            "and --block-angle/--block-angle-deg."
        )

    return [args.agent_x, args.agent_y, args.block_x, args.block_y, float(angle)]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zarr_path = resolve_zarr_path(args.zarr_path)
    states, actions, _ = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    model = build_policy(
        "mse",
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
        hidden_dims=tuple(args.hidden_dims),
    ).to(device)
    model.eval()

    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    reset_state = resolve_reset_state(args)
    reset_options = None
    if reset_state is not None:
        reset_options = {"reset_to_state": reset_state}
    obs, _ = env.reset(seed=args.seed, options=reset_options)
    initial_obs = obs.copy()

    frames = [env.render()]
    done = False
    step = 0
    max_reward = 0.0
    action_chunk: np.ndarray | None = None
    chunk_index = args.chunk_size

    while not done:
        if action_chunk is None or chunk_index >= args.chunk_size:
            state = torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
            with torch.no_grad():
                pred_chunk = model.sample_actions(state.unsqueeze(0)).cpu().numpy()[0]

            action_chunk = normalizer.denormalize_action(pred_chunk)
            action_chunk = np.clip(
                action_chunk,
                env.action_space.low,
                env.action_space.high,
            )
            chunk_index = 0

        action = action_chunk[chunk_index].astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())

        max_reward = max(max_reward, float(reward))
        done = terminated or truncated
        chunk_index += 1
        step += 1

    env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, fps=args.fps)

    print(f"Saved video: {args.output}")
    print(f"Initial observation: {initial_obs}")
    if reset_state is not None:
        print(f"Requested reset state: {reset_state}")
    print(f"Steps: {step}")
    print(f"Max reward: {max_reward:.4f}")


if __name__ == "__main__":
    main()
