# file_utils.py

import os
import json

def read_text_file(file_path, encoding='utf-8'):
    """
    Reads and returns the content of a text file.
    """
    with open(file_path, 'r', encoding=encoding) as file:
        return file.read()

def write_text_file(file_path, content, encoding='utf-8'):
    """
    Writes the given content to a text file.
    """
    with open(file_path, 'w', encoding=encoding) as file:
        file.write(content)

def read_json_file(file_path, encoding='utf-8'):
    """
    Reads and returns the content of a JSON file as a Python object.
    """
    with open(file_path, 'r', encoding=encoding) as file:
        return json.load(file)

def write_json_file(file_path, data, encoding='utf-8', indent=2):
    """
    Writes a Python object as JSON to a file.
    """
    with open(file_path, 'w', encoding=encoding) as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)

def create_directory(directory_path):
    """
    Creates a directory if it doesn't already exist.
    """
    os.makedirs(directory_path, exist_ok=True)

def get_file_basename(file_path):
    """
    Returns the base name of the file without the extension.
    """
    return os.path.splitext(os.path.basename(file_path))[0]

def get_absolute_path(file_path):
    """
    Returns the absolute path of the given file.
    """
    return os.path.abspath(file_path)
