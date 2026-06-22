import omni.ui as ui


class PlaybackUI:
    """
    Floating control panel inside the Isaac Sim GUI.
    Exposes pause/play, reset, and seek-to-frame controls.
    State is communicated back to the Simulator via attributes
    on the passed `simulator` object:
      _paused         : bool
      _reset_requested: bool
      _seek_frame     : int | None
    """

    def __init__(self, n_frames: int, simulator):
        self._sim = simulator
        self._n_frames = n_frames
        self._updating = False  # suppress seek callback while syncing slider

        self._window = ui.Window(
            "Replay Controls",
            width=320,
            height=200,
            flags=ui.WINDOW_FLAGS_NO_SCROLLBAR,
        )
        self._build()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build(self):
        with self._window.frame:
            with ui.VStack(spacing=8):
                # Frame counter
                with ui.HStack(height=24):
                    ui.Label("Frame:", width=70)
                    self._frame_label = ui.Label(f"0 / {self._n_frames - 1}")

                # Play / Pause  +  Reset
                with ui.HStack(spacing=6, height=30):
                    self._play_btn = ui.Button(
                        "Pause", width=100, clicked_fn=self._on_play_pause
                    )
                    ui.Button("Reset", width=80, clicked_fn=self._on_reset)

                # Seek slider
                with ui.HStack(height=24):
                    ui.Label("Seek:", width=70)
                    self._seek_model = ui.SimpleIntModel(0, min=0, max=max(0, self._n_frames - 1))
                    ui.IntSlider(model=self._seek_model)
                    self._seek_model.add_value_changed_fn(self._on_seek_changed)

                # Seek integer field (manual frame entry)
                with ui.HStack(height=24):
                    ui.Label("Go to frame:", width=70)
                    self._frame_field = ui.IntField(model=self._seek_model)

    # ------------------------------------------------------------------
    # Callbacks (called from the UI thread — same thread as the sim loop)
    # ------------------------------------------------------------------

    def _on_play_pause(self):
        self._sim._paused = not self._sim._paused
        self._play_btn.text = "Play" if self._sim._paused else "Pause"

    def _on_reset(self):
        self._sim._reset_requested = True
        self._sim._paused = False
        self._play_btn.text = "Pause"

    def _on_seek_changed(self, model):
        if not self._updating:
            self._sim._seek_frame = model.as_int

    # ------------------------------------------------------------------
    # Called by the sim loop each frame to keep the UI in sync
    # ------------------------------------------------------------------

    def update(self, frame: int):
        self._frame_label.text = f"{frame} / {self._n_frames - 1}"
        self._updating = True
        self._seek_model.set_value(frame)
        self._updating = False

    def destroy(self):
        self._window.destroy()
