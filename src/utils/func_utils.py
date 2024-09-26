import functools
import time


def retry(func):
    """Retry a function if it throws an exception."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        error = None
        for backoff in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Backoff and try again
                time.sleep(backoff)
                print(
                    f"Encountered error {e} while running {func.__name__}. Retrying {func.__name__} after {backoff} seconds."
                )
                error = e
                continue

        # If we get here, we've exhausted all retries
        print(
            f"Encountered error {error} while running {func.__name__}. Retries exhaused. Aborting."
        )
        raise error

    return wrapper
