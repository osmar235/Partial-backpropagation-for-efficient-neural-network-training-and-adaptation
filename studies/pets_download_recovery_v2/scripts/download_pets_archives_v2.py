"""Oxford download recovery with no overall transfer deadline.

Preserves v1 for provenance. Only the curl command changes: ignore user curlrc
defaults and explicitly disable the overall timeout. Connection and low-speed
timeouts, official hashes, extraction checks and partial preservation remain.
"""
import download_pets_archives as engine

original_command=engine.command

def command(curl,url,target,resume):
    args=original_command(curl,url,target,resume)
    # --disable must be the FIRST curl option to suppress a curlrc time limit.
    args.insert(1,'--disable')
    args[args.index('--max-time')+1]='0'
    return args

engine.command=command

if __name__=='__main__':
    print('Download recovery v2: no overall time limit. Only connection/stall timeouts apply.',flush=True)
    engine.main()
