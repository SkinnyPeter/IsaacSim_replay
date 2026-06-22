# Hugging Face dataset utilities

These scripts transfer either the older flat H5/trajectory layout or complete manipulation folders. Install the root requirements and authenticate with `hf auth login`, `HUGGINGFACE_TOKEN`, or `HF_TOKEN` in a project-root `.env` file.

```env
HUGGINGFACE_TOKEN=hf_your_token_here
```

The default repository ID is currently project-specific. Pass `--repo-id <owner/dataset>` where supported, or update the constants in scripts that do not expose that option.

## Recommended: upload one manipulation folder

Given this local structure:

```text
data/<manipulation-id>/
├── <recording>.h5
└── object_trajectory/*.npy
```

Preview and upload it with:

```bash
python scripts/huggingface/upload_manipulation_folder.py <manipulation-id> --repo-id <owner/dataset> --dry-run
python scripts/huggingface/upload_manipulation_folder.py <manipulation-id> --repo-id <owner/dataset>
```

Add `--exclude-h5` to omit H5/HDF5 files. Use `--data-root` for a non-default local root and `--path-in-repo` for a custom remote directory.

## Timestamp-only folder without H5

For IDs matching `YYYYMMDD_HHMMSS`:

```bash
python scripts/huggingface/upload_manipulation_without_h5.py <manipulation-id> --repo-id <owner/dataset> --dry-run
python scripts/huggingface/upload_manipulation_without_h5.py <manipulation-id> --repo-id <owner/dataset>
```

This command always excludes `.h5` and `.hdf5` files.

## Legacy flat layout

`download_from_id.py` and `upload_data_in_hf.py` target the older layout:

```text
data/
├── h5/<YYYYMMDD_HHMMSS>.h5
└── trajectories/<YYYYMMDD_HHMMSS>_trajectory.npy
```

```bash
python scripts/huggingface/download_from_id.py <YYYYMMDD_HHMMSS>
python scripts/huggingface/upload_data_in_hf.py
```

These two scripts use a hard-coded repository ID; inspect it before use. Always use a dry run where one is available, and confirm that no private recordings or credentials are included before uploading.
