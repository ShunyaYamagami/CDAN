import os
from glob import glob
from pathlib import Path
iter_model_dirs = glob('snapshot/CDAN+E/DomainNet/*')

for d in iter_model_dirs[:-1]:
    print(Path(d).name)
    iter_model_dirs = sorted(glob(os.path.join(d, 'iter_*.pth.tar')))
    for m in iter_model_dirs[:-1]:
        print(Path(m).name)
        os.remove(m)