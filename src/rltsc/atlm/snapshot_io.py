import logging


import os


import random


import time


from pathlib import Path


from typing import Any


from filelock import FileLock, Timeout


def dump_cityflow_snapshot(
    engine: Any,
    snapshot_path: Path,
    lock_path: Path,
    logger: logging.Logger,
    current_time: float,
    max_retries: int = 6,
    lock_timeout: float = 180.0,
) -> bool:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    lock = FileLock(lock_path.as_posix(), timeout=lock_timeout)

    for attempt in range(1, max_retries + 1):
        temp_path = snapshot_path.with_name(f".{snapshot_path.name}.{os.getpid()}.{time.time_ns()}.tmp")

        try:
            with lock:
                cityflow_state_snapshot = engine.snapshot()

                cityflow_state_snapshot.dump(temp_path.as_posix())

                if temp_path.exists() and temp_path.stat().st_size > 0:
                    temp_path.replace(snapshot_path)

                    return True

            logger.warning(
                "Empty CityFlow snapshot written on attempt %s/%s at time=%s: %s",
                attempt,
                max_retries,
                current_time,
                snapshot_path,
            )

        except Timeout:
            logger.warning(
                "Timed out waiting for snapshot dump lock on attempt %s/%s at time=%s: %s",
                attempt,
                max_retries,
                current_time,
                lock_path,
            )

        except Exception:
            logger.exception(
                "Failed to dump CityFlow snapshot on attempt %s/%s at time=%s: %s",
                attempt,
                max_retries,
                current_time,
                snapshot_path,
            )

        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

        sleep_seconds = min(3.0, 0.2 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.3)

        time.sleep(sleep_seconds)

    logger.error("Failed to write non-empty CityFlow snapshot after %s attempts: %s", max_retries, snapshot_path)

    return False
