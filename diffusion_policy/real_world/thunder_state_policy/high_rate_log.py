"""
Controller-rate (500 Hz) log of the OSC loop for checking a simulator's arm model against real policy runs.

RealEnv keeps only the last max_obs_buffer_size (30) controller samples in its ring buffer and the replay buffer stores one
sample per policy step, so neither shows what the arm does within a 0.1 s step. HighRateLog records every control loop:
the robot state the torque was computed from, the torque sent, the OSC target, and a few extra robot readings. The
control loop only copies numbers into a preallocated block; a background thread writes each full block (chunk_rows loops,
10 s at 500 Hz) to <directory>/chunk_NNNNN.npy, so file writes never run in the 2 ms loop and a crash loses at most one
block. load() returns the whole session as named arrays.

This is validation data (does a fitted model predict the arm in the regime the policy uses?), not a replacement for the
identification excitation: policy runs contain contacts and depend on the policy (PACE, Bjelonic et al. 2025).
"""
import json
import pathlib
import queue
import threading

import numpy as np

# name, width. Times are time.time() (host) and RTDE's controller timestamp (robot, s since controller start).
FIELDS = (
    ('t_host', 1),             # host time when q, qd were read for this loop
    ('t_robot', 1),            # robot controller timestamp of that state (getTimestamp)
    ('q', 6),                  # joint positions the torque was computed from (rad)
    ('qd', 6),                 # joint velocities the torque was computed from (rad/s)
    ('torque', 6),             # torque sent with directTorque (N m)
    ('target_pos', 3),         # OSC target position, base_link (m)
    ('target_quat', 4),        # OSC target orientation, base_link, wxyz
    ('current', 6),            # motor currents (A, getActualCurrent)
    ('tcp_force', 6),          # TCP wrench (getActualTCPForce)
    ('gripper_close_cmd', 1),  # 1 if the gripper is commanded closed
    ('gripper_position', 1),   # last Robotiq gPO read by GripperStateReader (NaN if not read)
    ('target_seq', 1),         # number of targets received so far: rows with the same value share one policy target
)
WIDTH = sum(w for _, w in FIELDS)
SLICES = {}
_start = 0
for _name, _w in FIELDS:
    SLICES[_name] = slice(_start, _start + _w)
    _start += _w


class HighRateLog:
    def __init__(self, directory, meta=None, chunk_rows=5000, spare_blocks=3):
        self.dir = pathlib.Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chunk_rows = int(chunk_rows)
        info = dict(meta or {}, fields=[[n, w] for n, w in FIELDS], chunk_rows=self.chunk_rows)
        (self.dir / 'meta.json').write_text(json.dumps(info, indent=1, default=str) + '\n')
        self.free = queue.Queue()
        for _ in range(spare_blocks):
            self.free.put(np.full((self.chunk_rows, WIDTH), np.nan))
        self.block = self.free.get()
        self.n = 0
        self.chunk = 0
        self.extra_blocks = 0
        self.errors = []
        self.pending = queue.Queue()
        self.thread = threading.Thread(target=self._writer, name='high-rate-log', daemon=True)
        self.thread.start()

    def append(self, **values):
        """One control loop. Missing fields stay NaN."""
        row = self.block[self.n]
        for name, value in values.items():
            row[SLICES[name]] = value
        self.n += 1
        if self.n == self.chunk_rows:
            self._hand_off()

    def _hand_off(self):
        if self.n == 0:
            return
        self.pending.put((self.chunk, self.block, self.n))
        self.chunk += 1
        try:
            self.block = self.free.get_nowait()
        except queue.Empty:                       # writer behind: allocate rather than block the control loop
            self.block = np.full((self.chunk_rows, WIDTH), np.nan)
            self.extra_blocks += 1                # an extra allocation; no data is lost
        self.n = 0

    def _writer(self):
        while True:
            item = self.pending.get()
            if item is None:
                return
            idx, block, n = item
            try:
                tmp = self.dir / f'chunk_{idx:05d}.tmp.npy'
                np.save(tmp, block[:n])
                tmp.rename(self.dir / f'chunk_{idx:05d}.npy')
            except Exception as exc:              # keep the control loop alive; report at close()
                self.errors.append(f'chunk {idx}: {type(exc).__name__}: {exc}')
            block[:] = np.nan
            self.free.put(block)

    def close(self, timeout=10.0):
        """Write the partial block and wait for the writer. Returns the list of write errors (empty if none)."""
        self._hand_off()
        self.pending.put(None)
        self.thread.join(timeout=timeout)
        if self.errors:
            (self.dir / 'write_errors.json').write_text(json.dumps(self.errors, indent=1) + '\n')
        return self.errors


def load(directory):
    """(arrays, meta): every field of every chunk in order, as {name: (N,) or (N, w)}."""
    directory = pathlib.Path(directory)
    meta = json.loads((directory / 'meta.json').read_text())
    chunks = sorted(p for p in directory.glob('chunk_*.npy') if not p.name.endswith('.tmp.npy'))
    data = np.concatenate([np.load(p) for p in chunks]) if chunks else np.zeros((0, WIDTH))
    out, start = {}, 0
    for name, w in meta['fields']:
        out[name] = data[:, start] if w == 1 else data[:, start:start + w]
        start += w
    return out, meta
