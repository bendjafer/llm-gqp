import json
import os
import difflib
import tempfile
from tqdm import tqdm
from descriptors import DESCRIPTORS


class Corpus:

    def __init__(self, path="results/graphs_translation.json"):
        self.path  = path
        self._data = {}

    # ── Private ───────────────────────────────────────────────────────────────

    def _read_file(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def _write_file(self):
        """Atomic write: dump to a sibling .tmp file, then rename.

        A crash or KeyboardInterrupt mid-write leaves the .tmp file behind
        rather than a half-written, corrupt JSON.
        """
        dir_ = os.path.dirname(self.path) or "."
        os.makedirs(dir_, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=True)
            os.replace(tmp_path, self.path)  # atomic on POSIX
        except Exception:
            os.unlink(tmp_path)              # clean up on failure
            raise

    def _missing(self, graphs, descriptor_names):
        return {
            name: G for name, G in graphs.items()
            if name not in self._data
            or any(fmt not in self._data[name] for fmt in descriptor_names)
        }

    # ── Public ────────────────────────────────────────────────────────────────

    def load(self, graphs=None, descriptor_names=None):
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            try:
                self._data = self._read_file()
                return self
            except (json.JSONDecodeError, OSError) as e:
                raise ValueError(f"Corrupted corpus at '{self.path}': {e}") from e

        if graphs is None or descriptor_names is None:
            raise FileNotFoundError(
                f"No corpus at '{self.path}'. Pass graphs and descriptor_names to build it."
            )

        return self.build(graphs, descriptor_names)

    def save(self):
        self._write_file()

    def build(self, graphs, descriptor_names):
        descriptor_names = list(descriptor_names)

        for fmt in descriptor_names:
            if fmt not in DESCRIPTORS:
                suggestions = difflib.get_close_matches(fmt, DESCRIPTORS.keys(), n=3, cutoff=0.4)
                hint = f" Did you mean: {suggestions}?" if suggestions else f" Valid keys: {list(DESCRIPTORS.keys())}"
                raise KeyError(f"Unknown descriptor '{fmt}'.{hint}")

        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            try:
                self._data = self._read_file()
            except (json.JSONDecodeError, OSError) as e:
                raise ValueError(f"Corrupted corpus at '{self.path}': {e}") from e

        missing = self._missing(graphs, descriptor_names)

        if missing:
            graph_bar = tqdm(missing.items(), desc="Building descriptors", unit="graph")
            for name, G in graph_bar:
                graph_bar.set_postfix(graph=name)
                new_formats = {}
                fmt_bar = tqdm(descriptor_names, desc=f"  {name}", unit="fmt", leave=False)
                for fmt in fmt_bar:
                    fmt_bar.set_postfix(fmt=fmt)
                    new_formats[fmt] = DESCRIPTORS[fmt](G)
                if name not in self._data:
                    self._data[name] = new_formats
                else:
                    self._data[name].update(new_formats)
            self._write_file()

        return self

    def get(self, graph_name, fmt=None):
        if graph_name not in self._data:
            suggestions = difflib.get_close_matches(graph_name, self._data.keys(), n=3, cutoff=0.4)
            hint = f" Did you mean: {suggestions}?" if suggestions else f" Available: {list(self._data.keys())}"
            raise KeyError(f"'{graph_name}' not found.{hint}")
        if fmt is None:
            return self._data[graph_name]
        if fmt not in self._data[graph_name]:
            raise KeyError(f"Format '{fmt}' not in '{graph_name}'. Available: {list(self._data[graph_name].keys())}")
        return self._data[graph_name][fmt]

    def subset(self, graph_names, descriptor_names):
        return {
            name: {fmt: self._data[name][fmt] for fmt in descriptor_names}
            for name in graph_names
        }

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __contains__(self, graph_name):
        return graph_name in self._data

    def __iter__(self):
        return iter(self._data.items())

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        graphs  = list(self._data.keys())
        formats = list(next(iter(self._data.values())).keys()) if self._data else []
        return f"Corpus(graphs={graphs}, formats={formats}, path='{self.path}')"
