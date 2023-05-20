
from zipfile import ZipFile


def extract_zip(input_zip) -> dict[str, bytes]:
    input_zip = ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}


def insert_zip(zip_path: str, files: dict[str, bytes]):
    with open(zip_path, 'w+') as file_created:  # file creation TODO Create file with ZipFile
        pass
    with ZipFile(zip_path, mode='w') as archive:
        for file_name, file_content in files.items():
            archive.writestr(file_name, file_content)
