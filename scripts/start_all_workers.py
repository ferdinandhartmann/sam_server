"""Helper script to start all workers without running the API server."""

from worker_manager import WorkerManager


def main() -> None:
    manager = WorkerManager()
    manager.start()
    print("Workers are running. Use CTRL+C to stop.")
    try:
        import signal
        import time

        signal.signal(signal.SIGINT, lambda *_: None)
        signal.signal(signal.SIGTERM, lambda *_: None)
        while True:
            time.sleep(1)
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
