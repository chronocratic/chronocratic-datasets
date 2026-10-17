Fix crashes and truncated cache files when many processes (e.g. SLURM jobs) prepare the same dataset at the same time: each cache write now uses its own temp file before an atomic rename.
