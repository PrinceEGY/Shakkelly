from pathlib import Path


def save_string_to_file(string, filename):
    file = Path(filename)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(string)


def load_string_from_file(filename):
    file = Path(filename)
    return file.read_text()
