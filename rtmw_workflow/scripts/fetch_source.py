"""Fetch the exact official source revision used by this workflow."""
import hashlib
import io
from pathlib import Path
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rtmw_workflow.runtime import HERE, UPSTREAM, REVISION, CONFIG


def main():
    if CONFIG.is_file() and (UPSTREAM/'mmpose/models/heads/coord_cls_heads/rtmw_head.py').is_file():
        print(f'Using {UPSTREAM}')
        return
    url = f'https://codeload.github.com/open-mmlab/mmpose/zip/{REVISION}'
    data = urllib.request.urlopen(url, timeout=120).read()
    root = (HERE/'vendor').resolve()
    root.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for entry in archive.infolist():
            target = (root/entry.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Unsafe path in upstream archive')
        archive.extractall(root)
    (root/'source_sha256.txt').write_text(hashlib.sha256(data).hexdigest()+'\n', encoding='ascii')
    print(f'Downloaded official MMPose {REVISION}')


if __name__ == '__main__':
    main()
