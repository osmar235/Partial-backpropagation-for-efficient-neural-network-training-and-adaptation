"""Recover Oxford Pet downloads without modifying the frozen experiment runner."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/pets_transfer/oxford-iiit-pet'
# Official torchvision archive digests; no substitute dataset or unchecked mirror.
RESOURCES=(('images.tar.gz','5c4f3ee8e5d25df40f4fd59a7f44e54c','images'),
           ('annotations.tar.gz','95a8c909bbe2e81eed6a22bccdf3f68f','annotations'))
HOSTS=('https://thor.robots.ox.ac.uk/pets/',
       'https://www.robots.ox.ac.uk/~vgg/data/pets/data/')

def md5(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'md5').hexdigest()

def preserve(path,reason):
    target=path.with_name(path.name+'.'+reason+'.'+uuid.uuid4().hex)
    assert path.resolve().parent==target.resolve().parent
    path.rename(target)
    print('Preserved '+str(target),flush=True)
    return target

def command(curl,url,target,resume):
    cmd=[curl,'--location','--fail','--proto','=https','--proto-redir','=https',
         '--http1.1','--connect-timeout','20','--speed-limit','1024','--speed-time','90',
         '--max-time','3600','--progress-bar','--output',str(target)]
    if resume:cmd+=['--continue-at','-']
    return cmd+[url]

def download(target,expected,urls,curl,attempts=8,run=subprocess.run,sleep=time.sleep):
    target=Path(target)
    if target.exists() and md5(target)==expected:
        print('Verified existing '+target.name,flush=True);return
    errors=[]
    for attempt in range(attempts):
        offset=target.stat().st_size if target.exists() else 0
        url=urls[attempt%len(urls)]
        print(f'DOWNLOAD {target.name}: attempt {attempt+1}/{attempts}, existing bytes {offset}',flush=True)
        result=run(command(curl,url,target,bool(offset)),check=False)
        # A checksum, not a transport exit code or file length, determines completion.
        if target.exists() and md5(target)==expected:
            print('CHECKSUM PASS '+target.name,flush=True);return
        errors.append(result.returncode)
        if result.returncode in (0,33,36):
            # 33/36: server cannot resume. Never truncate the partial in place.
            reason='checksum_failed' if result.returncode==0 else 'resume_unavailable'
            if target.exists():preserve(target,reason)
            print('Server did not provide a usable complete/ranged archive; next attempt starts a preserved fresh copy.',flush=True)
        elif result.returncode in (3,4,48,60,77):
            raise RuntimeError(f'curl configuration or TLS verification error {result.returncode}; no TLS bypass attempted')
        if attempt+1<attempts:sleep(min(20,2*(attempt+1)))
    raise RuntimeError(f'Download incomplete after {attempts} attempts (curl codes {errors}). Partial bytes retained. Run the same command later.')

def extract_verified(archive,expected,folder,base):
    base=Path(base).resolve();archive=Path(archive)
    if md5(archive)!=expected:raise ValueError('Refusing to extract an archive with the wrong checksum')
    stamp=base/(folder+'.verified_extraction.json');target=base/folder
    if stamp.exists() and target.is_dir() and json.loads(stamp.read_text())['archive_md5']==expected:
        print('Verified extraction already installed: '+folder,flush=True);return
    stage=base/('.extract_'+uuid.uuid4().hex);stage.mkdir()
    assert stage.resolve().parent==base and target.resolve().parent==base
    # Enforce relative paths, a single expected root, regular files/directories,
    # and no links/devices. Do not expose partially extracted final directories.
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        for member in members:
            rel=Path(member.name)
            if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0]!=folder:
                raise ValueError('Unexpected archive path: '+member.name)
            if not (member.isfile() or member.isdir()):raise ValueError('Archive link/device rejected: '+member.name)
            if not (stage/rel).resolve().is_relative_to(stage.resolve()):raise ValueError('Archive path escapes staging folder')
        tar.extractall(stage,members=members,filter='data')
    installed=stage/folder;assert installed.is_dir()
    if folder=='annotations':assert (installed/'trainval.txt').is_file()
    if target.exists():preserve(target,'pre_recovery')
    installed.rename(target)
    stage.rmdir()  # Empty staging directory only; failed stages remain preserved.
    stamp.write_text(json.dumps(dict(archive_md5=expected,source_archives='official Oxford',test_evaluation=False),indent=2)+'\n')
    print('EXTRACTED '+folder,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--attempts',type=int,default=8)
    parser.add_argument('--archive',choices=['all','images','annotations'],default='all')
    args=parser.parse_args();assert 2<=args.attempts<=30
    curl=shutil.which('curl.exe') or shutil.which('curl')
    if not curl:raise RuntimeError('curl is required; Windows includes curl.exe')
    BASE.mkdir(parents=True,exist_ok=True)
    # Same process lock as preparation, without importing torch or changing its sources.
    with (BASE.parent/'RUN.lock').open('a+b') as lock:
        lock.seek(0,2)
        if not lock.tell():lock.write(b'0');lock.flush()
        lock.seek(0)
        import os
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,expected,folder in RESOURCES:
            if args.archive!='all' and args.archive!=folder:continue
            archive=BASE/name
            download(archive,expected,[host+name for host in HOSTS],curl,args.attempts)
            extract_verified(archive,expected,folder,BASE)
    print('Official archives verified and extracted. Continue unchanged baseline preparation.',flush=True)

if __name__=='__main__':main()
