# mypy: allow-untyped-defs
import os.path
from glob import glob


__serialization_id_record_name__ = ".data/serialization_id"


class DirectoryReader:
    """
    Class to allow PackageImporter to operate on unzipped packages. Methods
    copy the behavior of the PackageFileReader class (which is used for
    accessing packages in all other cases).
    """

    def __init__(self, directory):
        self.directory = directory

    def get_record(self, name):
        filename = f"{self.directory}/{name}"
        with open(filename, "rb") as f:
            return f.read()

    def has_record(self, path):
        full_path = os.path.join(self.directory, path)
        return os.path.isfile(full_path)

    def get_all_records(
        self,
    ):
        files = [
            filename[len(self.directory) + 1 :]
            for filename in glob(f"{self.directory}/**", recursive=True)
            if not os.path.isdir(filename)
        ]
        return files

    def serialization_id(
        self,
    ):
        if self.has_record(__serialization_id_record_name__):
            return self.get_record(__serialization_id_record_name__)
        else:
            return ""
