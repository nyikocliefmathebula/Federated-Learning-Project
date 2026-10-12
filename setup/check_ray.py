# fl-iot-ids: Federated Learning-Based Intrusion Detection for Distributed IoT Networks
# Nyiko Mathebula, EEE4022S final-year project, University of Cape Town, 2026

"""Check that the installed Ray runtime can start a local CPU worker."""
# Ray is the engine Flower's simulation uses to run the simulated gateways.
# This starts Ray with 2 CPUs, asks one worker process to compute 41 + 1, and
# prints RAY_CHECK_OK if the answer 42 comes back within 60 seconds.
# Run: python setup/check_ray.py
import ray


# @ray.remote: this function runs in a separate Ray worker process
@ray.remote
def add_one(value):
    return value + 1


if __name__ == "__main__":
    # start a small local Ray instance (no GPU, no web dashboard)
    ray.init(num_cpus=2, num_gpus=0, include_dashboard=False)
    try:
        result = ray.get(add_one.remote(41), timeout=60)
        if result != 42:
            raise RuntimeError("Unexpected worker result.")
        print("RAY_CHECK_OK")
    finally:
        # always stop Ray again, also after an error
        ray.shutdown()
