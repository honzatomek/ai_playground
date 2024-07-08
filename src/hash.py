import hashlib

def hash_file(filename, hash_algorithm='sha256'):
    """
    Generates the hash of a file.

    Parameters:
        filename (str): Path to the file.
        hash_algorithm (str): Hashing algorithm (default is 'sha256').

    Returns:
        str: The hexadecimal hash of the file.
    """
    hash_func = getattr(hashlib, hash_algorithm)()

    with open(filename, 'rb') as file:
        while chunk := file.read(8192):  # Read the file in 8KB chunks
            hash_func.update(chunk)

    return str(hash_func.hexdigest())

