import argparse
import logging
from pathlib import Path
from datetime import datetime

from src.config import load_config, LogConfig
from isaacsim import SimulationApp

LOG_FORMAT = "%(levelname)-8s %(name)s — %(message)s"

BASE_DIR = Path(__file__).resolve().parent


def _check_paths(scene_path, data_path):
    missing = []
    for label, path in [("Scene", scene_path), ("Data", data_path)]:
        if not path.exists():
            missing.append(f"  {label}: {path}")
    if missing:
        raise FileNotFoundError(
            "Missing required files — fix before Isaac Sim starts:\n" + "\n".join(missing)
        )


def _setup_logging(log_cfg: LogConfig):
    console_level = getattr(logging, log_cfg.console_level.upper(), logging.INFO)
    logging.basicConfig(level=console_level, format=LOG_FORMAT)

    if log_cfg.file_enabled:
        log_dir = BASE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        src_logger = logging.getLogger("src")
        src_logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_cfg.file_level.upper(), logging.DEBUG))
        src_logger.addHandler(file_handler)


def main():
    parser = argparse.ArgumentParser(description="Replay world-space object trajectories in Isaac Sim")
    parser.add_argument("manip_id", help="Manipulation ID, e.g. 20250804_104715")
    args = parser.parse_args()

    scene_path, data_path, robot_cfg, sim_cfg, vis_cfg, seg_cfg, log_cfg, rec_cfg = load_config(
        BASE_DIR / "config" / "world_traj_config.yaml", base_dir=BASE_DIR,
        manipulation_id=args.manip_id,
    )
    _check_paths(scene_path, data_path)
    _setup_logging(log_cfg)

    simulation_app = SimulationApp({"headless": sim_cfg.headless})

    from src.simulator.world_traj.simulator import WorldTrajectorySimulator

    simulator = WorldTrajectorySimulator(
        simulation_app, scene_path, data_path,
        sim_config=sim_cfg, vis_config=vis_cfg, robot_config=robot_cfg,
        seg_config=seg_cfg, recording_config=rec_cfg,
    )
    simulator.play()

    simulation_app.close()


if __name__ == "__main__":
    main()
