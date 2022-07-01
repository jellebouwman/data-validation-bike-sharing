import yaml


def load_params(params_path: str):
    """Loads configuration from yaml file.

    Args:
        params_path: Filepath to the configuration.

    Returns:
        Dictionary containing configuration.
    """
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params