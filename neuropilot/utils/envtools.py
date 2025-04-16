import os

# Key name used for job ID
JOB_ID_ENV_VAR = "NEUROPILOT_JOB_ID"

def set_job_env(job: dict, env: dict = None) -> dict:
    """
    Returns a new environment dictionary with the job ID injected.

    Args:
        job (dict): The job dictionary (must include 'job_id')
        env (dict): Optional existing environment to copy (e.g., os.environ)

    Returns:
        dict: New environment with the job ID set
    """
    env_copy = dict(env) if env else dict(os.environ)
    job_id = job.get("job_id")

    if not job_id:
        raise ValueError("Job must include a 'job_id' key to inject into environment.")

    env_copy[JOB_ID_ENV_VAR] = job_id
    return env_copy


def get_job_id_from_env(default=None) -> str:
    """
    Reads the current job ID from environment, if set.

    Args:
        default: Value to return if not set (default: None)

    Returns:
        str or None: The job ID string, or fallback if not set
    """
    return os.environ.get(JOB_ID_ENV_VAR, default)


def is_job_id_set() -> bool:
    """
    Returns True if a job ID is set in the environment.

    Returns:
        bool: Whether the job ID environment variable is set
    """
    return JOB_ID_ENV_VAR in os.environ
