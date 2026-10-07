"""Download and verify the versioned dataset from this repository's GitHub Release."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def extract_verified(archive,manifest,destination):
    """Extract only the explicitly registered files; reject traversal and duplicates."""
    destination=Path(destination).resolve()
    records=[r for r in manifest['files'] if r.get('in_archive')]
    expected={r.get('archive_path',r['path']):r for r in records}
    if len(expected)!=len(records):
        raise ValueError('Duplicate archive paths in manifest')
    targets=[(destination/r['path']).resolve() for r in records]
    if len(set(targets))!=len(targets):
        raise ValueError('Duplicate installation paths in manifest')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(expected):
            raise ValueError('Archive members do not match the release manifest')
        for name in names:
            record=expected[name]
            target=(destination/record['path']).resolve()
            if not (destination/name).resolve().is_relative_to(destination) or not target.is_relative_to(destination):
                raise ValueError('Archive path escapes destination')
            if z.getinfo(name).file_size!=record['bytes']:
                raise ValueError(f'Archive member size mismatch: {name}')
            if target.is_file() and digest(target)==record['sha256']:
                print(f'Already verified: {name}',flush=True)
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.download')
            with z.open(name) as src,temporary.open('wb') as out:
                shutil.copyfileobj(src,out,8*1024*1024)
            if digest(temporary)!=record['sha256']:
                temporary.unlink()
                raise ValueError(f'Extracted file checksum mismatch: {name}')
            temporary.replace(target)
            print(f'Installed: {name}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,help='Use an already downloaded official ZIP')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    record=manifest['archive']
    archive=args.archive
    if archive is None:
        archive=ROOT/'.downloads'/record['name']
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists() or digest(archive)!=record['sha256']:
            partial=archive.with_suffix('.part')
            request=urllib.request.Request(record['url'],headers={'User-Agent':'DAS-Perimeter-Benchmark'})
            completed=0
            with urllib.request.urlopen(request,timeout=90) as response,partial.open('wb') as out:
                while chunk:=response.read(8*1024*1024):
                    out.write(chunk);completed+=len(chunk)
                    if completed>record['bytes']:
                        raise ValueError('Download exceeds the registered archive size')
                    print(f'\rDownloaded {completed/1e6:.0f}/{record["bytes"]/1e6:.0f} MB',end='',flush=True)
            print()
            if partial.stat().st_size!=record['bytes'] or digest(partial)!=record['sha256']:
                raise ValueError('Archive checksum mismatch; the partial download was not installed')
            partial.replace(archive)
    if archive.stat().st_size!=record['bytes'] or digest(archive)!=record['sha256']:
        raise ValueError('The supplied archive does not match the versioned release')
    extract_verified(archive,manifest,ROOT)
    print('Dataset ready. Run python scripts/verify_data.py for a full schema check.')


if __name__=='__main__':
    main()
