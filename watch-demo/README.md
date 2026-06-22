# H5 inspection and playback

These two lightweight tools inspect demonstrations without starting Isaac Sim. Install the root `requirements.txt`, then run them from the repository root with a normal Python environment.

## Interactive camera player

```bash
python watch-demo/play_h5_video.py data/<manipulation-id>/<recording>.h5
python watch-demo/play_h5_video.py <file.h5> --camera aria --camera oakd
python watch-demo/play_h5_video.py <file.h5> --start 100 --end 500 --fps 15
```

If no path is supplied, the player selects the lexically latest `.h5` under `data/h5/`. Camera arguments may be the aliases `aria` and `oakd` or full H5 dataset paths.

Controls:

- `Space`: pause or resume
- `+` / `-`: change playback speed
- `,` / `.` or left/right arrows: step one frame while paused
- Trackbar: seek
- `Q` or `Esc`: quit

## Structure and trajectory inspector

```bash
python watch-demo/inspect_h5.py <file.h5>
python watch-demo/inspect_h5.py <file.h5> --preview
python watch-demo/inspect_h5.py <file.h5> --plot-joints
python watch-demo/inspect_h5.py <file.h5> --plot-joints --key observations/qpos_arm --max-frames 1000
```

The default view prints groups, dataset shapes, and dtypes. `--preview` includes a short first-row preview. `--plot-joints` opens Matplotlib plots for discovered joint/action datasets or for datasets selected with repeated `--key` arguments.
