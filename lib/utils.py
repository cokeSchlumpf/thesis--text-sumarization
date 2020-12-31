from pathlib import Path


def root_directory(path='.') -> str:
    """
    Returns the projects root-directory, determined by the existence of the `.git` directory.

    :param path: The path to start the search. Default is current directory.
    :return:
    """
    file = Path(f"{path}/.git")

    if file.is_dir():
        return path
    else:
        return root_directory(f"{path}/..")

