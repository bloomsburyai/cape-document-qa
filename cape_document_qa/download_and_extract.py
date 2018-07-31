import math
from shutil import copyfileobj, unpack_archive, register_unpack_format, move, unregister_unpack_format
from hashlib import sha256
from pathlib import Path
import gzip
import os
import zipfile
from typing import Optional
from retry import retry
import requests
from tqdm import tqdm
from logging import info


def unpack_gzip(archive_path: str, destination_dir: str):
    destination_filepath = os.path.join(destination_dir, Path(archive_path).stem)
    with gzip.open(archive_path, 'rb') as f_in:
        with open(destination_filepath, 'wb') as f_out:
            copyfileobj(f_in, f_out)


def unpack_zip(archive_path: str, destination_dir: str):
    zfile = zipfile.ZipFile(archive_path)
    zfile.extractall(destination_dir)


register_unpack_format('gzip', ['.gz'], unpack_gzip)
unregister_unpack_format('zip')
register_unpack_format('zip', ['.zip'], unpack_zip)


@retry(delay=1, backoff=2, tries=4)
def download_and_extract(url: str, destination_folder: str, total_mb_size: Optional[float] = None) -> bool:
    """Download and extract from url, if file has already been downloaded return False else True."""
    try:
        # another process is currently (or was) working on the same url and destination folder
        key = url + destination_folder
        marker_filepath = os.path.join(destination_folder, sha256(key.encode()).hexdigest() + '.marker')
        Path(destination_folder).mkdir(exist_ok=True)
        Path(marker_filepath).touch(exist_ok=False)
    except FileExistsError:
        return False
    try:
        filepath = os.path.join(destination_folder, Path(url).name)
        download_or_resume(url, filepath, total_mb_size)
        unpack_archive(filepath, destination_folder)  # extract to destination
        os.remove(filepath)
    except Exception:
        os.remove(marker_filepath)
        info(f"Could not download {url}, retrying or aborting...")
        raise
    except (KeyboardInterrupt, SystemExit):
        os.remove(marker_filepath)
        info(f"Aborting download for system exit...")
        raise
    return True


def download_or_resume(url, file_path, total_mb_size=None):
    block_size = 1000 * 1000
    tmp_file_path = file_path + '.part'
    first_byte = os.path.getsize(tmp_file_path) if os.path.exists(tmp_file_path) else 0
    info(f'Starting download {url} at %.1fMB' % (first_byte / 1e6))
    file_size = int(requests.head(url).headers.get('Content-length', -1))
    if total_mb_size is None:
        if file_size != -1:
            total_mb_size = math.ceil(file_size / block_size)
    else:
        total_mb_size = math.ceil(total_mb_size)
    response = requests.get(url, headers={"Range": "bytes=%s-" % first_byte}, stream=True)
    with open(tmp_file_path, 'ab' if first_byte else 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=block_size), total=total_mb_size, unit='MB',
                          initial=first_byte // 1e6):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    final_size = os.path.getsize(tmp_file_path)
    info(f"Downloaded {final_size//block_size} MB")
    if file_size == -1 or file_size == final_size:
        move(tmp_file_path, file_path)
