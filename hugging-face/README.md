# Legacy Hugging Face pipeline helpers

This directory contains the older, pipeline-output-oriented Hugging Face scripts. They use the fixed dataset repository configured in each script and expect initial poses plus state CSV files. For uploading an entire current `data/<manipulation-id>/` folder, prefer the tools documented in `scripts/huggingface/README.md`.

## Authentication

Install the root requirements, then either log in with the Hugging Face CLI or create `hugging-face/.env`:

```env
HUGGINGFACE_TOKEN=hf_your_token_here
```

The `.env` file is ignored by Git. Never commit it.

```bash
python hugging-face/test_hf_login.py
```

## Expected source layout

`upload_data_in_hf.py` expects:

```text
<source-dir>/
├── object_initial_pose.npy
├── container_initial_pose.npy
├── object_states.csv
├── container_states.csv
├── about.yaml                       # created when absent
├── object_trajectory.npy            # optional
└── container_trajectory.npy         # optional
```

Upload one manipulation:

```bash
python hugging-face/upload_data_in_hf.py <manipulation-id> --source-dir <source-dir>
```

Use `--object` and `--container` to override the metadata names. The script writes files below `<manipulation-id>/` in the configured dataset repository.

`download_from_id.py` mirrors the older fixed files `trajectory_yellow_rubber_duck.npy`, `trajectory_purple_bowl.npy`, and `about.yaml`:

```bash
python hugging-face/download_from_id.py <YYYYMMDD_HHMMSS>
```

Review the hard-coded repository ID and file names in these scripts before using them in another project.
