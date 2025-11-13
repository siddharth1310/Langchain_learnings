import os
import time
import requests
from functools import wraps
from logger_setup import get_logger
from openai import APIConnectionError, APITimeoutError, RateLimitError

# Get a logger for this file
logger = get_logger(__file__)

def validate_env_vars(required_vars : list[str]):
    """
    Validates that all required environment variables are present and loaded.

    <b>*Parameters*</b>
    - required_vars (List[str]): 
        A list of environment variable names (as strings) that are essential 
        for the application to run correctly — e.g., API keys, database URLs, etc.

    <b>*Raises*</b>
    - EnvironmentError: 
        If one or more required environment variables are missing or not set.

    <b>*Logic*</b>
    1. Iterate through all variable names in the `required_vars` list.
    2. Check whether each variable exists in the environment using `os.getenv(var)`.
    3. Collect any missing variable names into a list.
    4. If the list of missing variables is not empty, raise an informative error 
       specifying which variables are missing — helping developers debug setup issues quickly.
    """
    
    # Identify which of the required environment variables are missing.
    missing = [var for var in required_vars if not os.getenv(var)]

    # If any required environment variables are missing, raise an explicit error.
    # This ensures that configuration issues are caught early, before runtime failures occur.
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

# -------------------------------------------------------------------
# Define a global tuple of transient (temporary) errors worth retrying
# These are the errors that can succeed on another attempt — e.g.,
# network issues, API timeouts, or rate-limit throttling.
# -------------------------------------------------------------------
TRANSIENT_ERRORS = (
    requests.ConnectionError,  # Generic network disconnects
    requests.Timeout,          # Requests taking too long
    ConnectionResetError,      # TCP connection dropped/reset
    APIConnectionError,        # OpenAI-specific network failures
    APITimeoutError,           # OpenAI timeout issues
    RateLimitError,            # Too many API calls (rate-limited)
)

# -------------------------------------------------------------------
# Decorator: with_retry
# Retries a function multiple times if transient (temporary) errors occur.
# -------------------------------------------------------------------
def with_retry(max_attempts : int = 3, delay : int = 3, retry_exceptions : tuple = TRANSIENT_ERRORS):
    """
    A decorator that automatically retries a function call if it fails
    due to transient errors like connection issues or rate limits.

    Args:
        max_attempts (int): Maximum number of retry attempts before giving up.
        delay (int): Wait time (in seconds) between retry attempts.
        retry_exceptions (tuple): Error types considered transient/retryable.

    Returns:
        Wrapper function that applies retry logic around the target function.
    """
    def decorator(fn):
        # wraps() keeps original metadata (name, docstring, etc.) for better logs
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Attempt the function up to 'max_attempts' times
            for attempt in range(1, max_attempts + 1):
                try:
                    # Try running the wrapped function
                    result = fn(*args, **kwargs)
                    
                    # ✅ If no exception, log success and return immediately
                    logger.info(f"✅ {fn.__name__} succeeded on attempt {attempt}")
                    return result

                # --------------------------------------------------------
                # Handle retryable (transient) exceptions
                # --------------------------------------------------------
                except retry_exceptions as e:
                    # Warn that this attempt failed but might work next time
                    logger.warning(f"⚠️ {fn.__name__} attempt {attempt} failed due to transient error: {e}")
                    
                    # If we still have attempts left → wait and retry
                    if attempt < max_attempts:
                        logger.info(f"⏳ Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        # ❌ All attempts used — log and re-raise the last error
                        logger.error(f"⛔ {fn.__name__} failed after {max_attempts} attempts.")
                        raise
                
                # --------------------------------------------------------
                # Handle non-retryable exceptions (e.g., logic bugs, invalid inputs)
                # --------------------------------------------------------
                except Exception as e:
                    # Log error and stop retrying immediately
                    logger.error(f"❌ Non-retryable error in {fn.__name__}: {e}")
                    raise
        return wrapper
    return decorator

