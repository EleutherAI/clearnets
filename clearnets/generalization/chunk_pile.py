# Split the pile into 8 billion token chunks named after roughly log spaced pythia training steps 
# Chunks are used to traing SAEs on the corresponding pythia checkpoint

from numpy import memmap
import os
from pathlib import Path
import math

log_spaced_checkpoints = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1000,
    2000,
    4000,
    8000,
    16_000,
    33_000,
    66_000,
    131_000,
    143_000,
]

pile_path = '/mnt/ssd-1/pile_preshuffled/deduped/document.bin'

# Get number of uint16s that can fit in this file by its file size
pile_size = os.path.getsize(pile_path) // 2
seq_len = 2049

pile_mmap = memmap(pile_path, dtype='uint16', mode='r', shape=(pile_size // seq_len, seq_len))

slice_path = Path('/mnt/ssd-1/lucia/pile_slices')
slice_path.mkdir(exist_ok=True)

num_samples = math.ceil(8_000_000_000 / 2049)

for i, step in enumerate(log_spaced_checkpoints):
    slice_mmap = memmap(
        slice_path / f'{step}.bin', 
        dtype='uint16', mode='w+', 
        shape=(num_samples, seq_len)
    )

    start = i * num_samples
    slice_mmap[:] = pile_mmap[start:start + num_samples]

    print(os.path.getsize(slice_path / f'{step}.bin') // 2)




