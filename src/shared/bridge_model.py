"""
Bridge Sensor Data Model  (v3)
==============================
Manages accelerometer sensors across bridge spans, merging spatial
positions with sigma-based noise thresholds loaded from two CSV files.

Adaptable to any bridge -- all bridge-specific info lives in the CSVs.
Axis direction is read from CSV, never inferred from the sensor ID.

Thresholds operate on |value| (absolute value) so the threshold is a
single positive number per sigma level, sign does not matter.

WHY EACH OOP CONCEPT IS USED
-----------------------------

1. @dataclass  (ThresholdLevel, Sensor)
   Sensors and thresholds are plain data carriers with a few helper
   methods.  @dataclass removes __init__ / __repr__ boilerplate and
   makes intent clear: these are structured records, not behaviour-
   heavy objects.

2. Composition  (Bridge -> Span -> Sensor -> ThresholdConfig)
   A bridge *has* spans, a span *has* sensors, a sensor *has* thresholds.
   This mirrors the physical hierarchy.  Each class owns only its own
   data, so you can test or reuse Sensor without needing Bridge.

   Why not inheritance?  A Sensor is not a "kind of Span".  The
   relationship is containment, not specialisation.

3. dict-based registry in Bridge  (_registry: dict[str, Sensor])
   With 100+ sensors, the most frequent operation is "give me sensor X
   by its ID".  A dict gives O(1) lookup vs O(n) scanning a list.
   This is the primary access pattern during processing.

4. Sorted list inside Span  (_sensors sorted by distance)
   Within a single span, we often need sensors in spatial order (for
   plotting, interpolation, waterfall plots).  A sorted list lets us
   iterate front-to-back naturally.  The list is small per span so
   sorting cost is negligible.

5. Multimap index in Bridge  (_loc_index: dict[tuple, list])
   Some sensors share the same (span, distance) because they sit on
   opposite sides of the bridge.  A dict mapping (span_id, distance)
   to a *list* of sensors handles this naturally -- one key, multiple
   values.

6. Factory function  (load_bridge)
   CSV parsing is messy (column names vary, types need conversion,
   two files must be merged).  Keeping that logic in a standalone
   function means Bridge/Sensor stay clean domain objects, and the
   factory can be swapped for a different loader (database, API, ...)
   without touching the model classes.

7. Column-name remapping  (pos_cols / thr_cols dicts)
   Different bridges may use different CSV headers ("distanza" vs
   "relative_distance").  Remapping via a dict avoids hardcoding any
   particular bridge's naming convention.

8. extra dict on Sensor
   CSVs may contain columns we did not anticipate.  Rather than
   ignoring them, we capture them in sensor.extra so no information
   is silently lost.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Union, Any
from collections import defaultdict
import warnings


# ── helpers ───────────────────────────────────────────────────────────────

def _safe_float(val: str, default: float = 0.0) -> float:
    """Parse a float from CSV, returning *default* on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _read_csv(path: Union[str, Path], delimiter: str = ",") -> List[Dict[str, str]]:
    """Read a CSV into a list of row-dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, "r", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


# ── ThresholdLevel ────────────────────────────────────────────────────────
#
# @dataclass because it is a simple data record (sigma, threshold value,
# baseline stats).  The only behaviour is the noise check which compares
# |value| against the threshold.

@dataclass
class ThresholdLevel:
    """
    Single sigma-based threshold.

    Because we apply thresholds to |value| (absolute value of the
    acceleration), each level is just one positive number:

        threshold = mean_abs + n * std

    A reading is noise if  |reading| <= threshold.
    A reading is signal if |reading| >  threshold.

    No lower bound is needed -- the absolute value is always >= 0.
    """
    sigma: int            # the n in  mean + n*std  (1, 2, or 3)
    threshold: float      # the cutoff applied to |value|
    mean_abs: float = 0.0 # mean of |values|
    std: float = 0.0      # std of |values|

    def is_noise(self, value: float) -> bool:
        """True if |value| is at or below the threshold (= noise)."""
        return abs(value) <= self.threshold

    def is_signal(self, value: float) -> bool:
        """True if |value| exceeds the threshold (= real vibration)."""
        return abs(value) > self.threshold

    def __repr__(self) -> str:
        return f"{self.sigma}sigma(threshold={self.threshold:.6f})"


# ── ThresholdConfig ───────────────────────────────────────────────────────
#
# @dataclass wrapping a dict[int -> ThresholdLevel].
#
# Why a dict keyed by sigma multiplier?
#   - O(1) to get the 2sigma level directly, no scanning.
#   - Extensible: if you later add 4sigma, nothing changes structurally.

@dataclass
class ThresholdConfig:
    """All sigma levels for one sensor."""
    _levels: Dict[int, ThresholdLevel] = field(default_factory=dict)

    def add(self, sigma: int, threshold: float,
            mean_abs: float = 0.0, std: float = 0.0) -> None:
        """Register a threshold level (e.g. sigma=2, threshold=0.0065)."""
        self._levels[sigma] = ThresholdLevel(sigma, threshold, mean_abs, std)

    def get(self, sigma: int) -> Optional[ThresholdLevel]:
        """Get a specific sigma level, or None."""
        return self._levels.get(sigma)

    @property
    def sigmas(self) -> List[int]:
        """Available sigma levels, sorted [1, 2, 3]."""
        return sorted(self._levels)

    @property
    def has_any(self) -> bool:
        return bool(self._levels)

    def is_noise(self, value: float, sigma: int = 1) -> bool:
        """Is |value| within the noise band at this sigma level?"""
        lvl = self._levels.get(sigma)
        if lvl is None:
            raise KeyError(f"sigma={sigma} not configured (available: {self.sigmas})")
        return lvl.is_noise(value)

    def filter_signal(self, values: list, sigma: int = 1) -> list:
        """Keep only readings whose |value| exceeds the threshold."""
        lvl = self._levels.get(sigma)
        if lvl is None:
            raise KeyError(f"sigma={sigma} not configured")
        return [v for v in values if lvl.is_signal(v)]

    def to_dict(self) -> dict:
        return {str(k): asdict(v) for k, v in self._levels.items()}

    def __repr__(self) -> str:
        parts = []
        for s in self.sigmas:
            parts.append(f"{s}sigma={self._levels[s].threshold:.6f}")
        return f"Thresholds({', '.join(parts)})"


# ── Sensor ────────────────────────────────────────────────────────────────
#
# @dataclass because a sensor is fundamentally a data record:
#   identity (sensor_id) + position (span, distance, side) + config (thresholds)
#
# Composition: a Sensor *has* a ThresholdConfig rather than inheriting
# from it.  This keeps threshold logic separate and testable.

@dataclass
class Sensor:
    """
    One accelerometer channel on the bridge.
    Every attribute comes from the CSV -- nothing is guessed from the ID.
    """
    sensor_id: str
    span_id: str
    relative_distance: float
    number: int = -1        # human-friendly number (e.g. 99), -1 = not assigned
    side: str = ""          # e.g. "left", "right" -- free-form from CSV
    axis: str = ""          # e.g. "x", "z" -- from CSV, NOT from sensor ID
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def location_key(self) -> Tuple[str, float]:
        """(span_id, distance) for grouping co-located sensors."""
        return (self.span_id, self.relative_distance)

    def shares_position_with(self, other: Sensor) -> bool:
        """True if same span + same distance but different sensor (opposite side)."""
        return (self.span_id == other.span_id
                and abs(self.relative_distance - other.relative_distance) < 1e-6
                and self.sensor_id != other.sensor_id)

    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id, "number": self.number,
            "span_id": self.span_id,
            "relative_distance": self.relative_distance,
            "side": self.side, "axis": self.axis,
            "thresholds": self.thresholds.to_dict(), "extra": self.extra,
        }

    def __repr__(self) -> str:
        parts = [f"'{self.sensor_id}'"]
        if self.number >= 0:
            parts.append(f"#{self.number}")
        parts.append(f"span='{self.span_id}'")
        parts.append(f"dist={self.relative_distance:.3f}")
        if self.side:
            parts.append(f"side={self.side}")
        if self.axis:
            parts.append(f"axis={self.axis}")
        return f"Sensor({', '.join(parts)})"


# ── Span ──────────────────────────────────────────────────────────────────
#
# Regular class (not dataclass) because it has mutable internal state
# (lazy-sorted sensor list) that dataclass conventions would fight.
#
# Why a sorted list?  Within a span the natural traversal is by
# position along the bridge (for waterfall plots, spatial analysis).
# Sorting is deferred until first access to avoid sorting on every add.

class Span:
    """One bridge span.  Sensors kept sorted by relative distance."""

    def __init__(self, span_id: str, length: Optional[float] = None):
        self.span_id = span_id
        self.length = length
        self._sensors: List[Sensor] = []
        self._dirty = False

    def add_sensor(self, sensor: Sensor) -> None:
        self._sensors.append(sensor)
        self._dirty = True

    def _sort(self) -> None:
        if self._dirty:
            self._sensors.sort(key=lambda s: (s.relative_distance, s.side))
            self._dirty = False

    @property
    def sensors(self) -> List[Sensor]:
        self._sort()
        return list(self._sensors)

    @property
    def sensor_ids(self) -> List[str]:
        self._sort()
        return [s.sensor_id for s in self._sensors]

    @property
    def unique_distances(self) -> List[float]:
        return sorted({s.relative_distance for s in self._sensors})

    def at_distance(self, distance: float, tol: float = 1e-6) -> List[Sensor]:
        """All sensors at a given distance (both sides of the bridge)."""
        return [s for s in self._sensors
                if abs(s.relative_distance - distance) < tol]

    def by_side(self, side: str) -> List[Sensor]:
        self._sort()
        return [s for s in self._sensors if s.side == side]

    def paired(self) -> Dict[float, List[Sensor]]:
        """Positions where >1 sensor sits (opposite-side pairs)."""
        groups: Dict[float, List[Sensor]] = defaultdict(list)
        for s in self._sensors:
            groups[s.relative_distance].append(s)
        return {d: ss for d, ss in groups.items() if len(ss) > 1}

    def __len__(self) -> int:
        return len(self._sensors)

    def __iter__(self) -> Iterator[Sensor]:
        self._sort()
        return iter(self._sensors)

    def __repr__(self) -> str:
        return f"Span('{self.span_id}', sensors={len(self)})"


# ── Bridge ────────────────────────────────────────────────────────────────
#
# Regular class because it manages three internal data structures:
#   _registry   : dict[sensor_id -> Sensor]          for O(1) ID lookup
#   _spans      : dict[span_id   -> Span]            for O(1) span lookup
#   _loc_index  : dict[(span,dist) -> list[Sensor]]  for spatial queries
#
# Why three?  Each serves a different access pattern.  Keeping them in
# sync is Bridge's responsibility (single point of truth).

class Bridge:
    """
    Top-level container.

    bridge["030911D2_x"]            -> Sensor  (O(1) by ID)
    bridge[99]                      -> Sensor  (O(1) by number)
    bridge.span("campata_2")        -> Span    (O(1))
    bridge.at("campata_2", 14.0)    -> [Sensor, ...]
    for span in bridge: ...         -> iterate spans
    "030911D2_x" in bridge          -> membership by ID
    99 in bridge                    -> membership by number
    """

    def __init__(self, name: str = "Bridge"):
        self.name = name
        self._spans: Dict[str, Span] = {}
        self._registry: Dict[str, Sensor] = {}
        self._num_registry: Dict[int, Sensor] = {}   # number -> Sensor
        self._loc_index: Dict[Tuple, List[Sensor]] = defaultdict(list)
        self._loc_dirty = False

    # --- registration ---
    def add(self, sensor: Sensor) -> None:
        if sensor.sensor_id in self._registry:
            warnings.warn(f"Overwriting sensor '{sensor.sensor_id}'")
        if sensor.span_id not in self._spans:
            self._spans[sensor.span_id] = Span(sensor.span_id)
        self._spans[sensor.span_id].add_sensor(sensor)
        self._registry[sensor.sensor_id] = sensor
        if sensor.number >= 0:
            self._num_registry[sensor.number] = sensor
        self._loc_dirty = True

    def _rebuild_loc(self) -> None:
        self._loc_index.clear()
        for s in self._registry.values():
            self._loc_index[s.location_key].append(s)
        self._loc_dirty = False

    # --- lookups ---
    def __getitem__(self, key: Union[str, int]) -> Sensor:
        """Lookup by sensor_id (str) or sensor number (int)."""
        if isinstance(key, int):
            return self._num_registry[key]
        return self._registry[key]

    def get(self, key: Union[str, int]) -> Optional[Sensor]:
        """Lookup by sensor_id or number, returns None if missing."""
        if isinstance(key, int):
            return self._num_registry.get(key)
        return self._registry.get(key)

    def resolve(self, keys: List[Union[str, int]]) -> List[Sensor]:
        """
        Resolve a mixed list of sensor IDs and/or numbers to Sensor objects.

        Example:
            bridge.resolve([99, "030911D2_x", 101])
            -> [Sensor(...), Sensor(...), Sensor(...)]

        Raises KeyError if any key is not found.
        """
        result = []
        for k in keys:
            s = self[k]  # uses __getitem__, raises KeyError on miss
            result.append(s)
        return result

    def resolve_ids(self, keys: List[Union[str, int]]) -> List[str]:
        """Resolve numbers/IDs to a list of sensor_id strings."""
        return [s.sensor_id for s in self.resolve(keys)]

    def resolve_numbers(self, keys: List[Union[str, int]]) -> List[int]:
        """Resolve IDs/numbers to a list of sensor numbers."""
        return [s.number for s in self.resolve(keys)]

    def span(self, span_id: str) -> Optional[Span]:
        return self._spans.get(span_id)

    def at(self, span_id: str, distance: float, tol: float = 1e-6) -> List[Sensor]:
        """All sensors at (span, distance), handles paired sensors."""
        if self._loc_dirty:
            self._rebuild_loc()
        exact = self._loc_index.get((span_id, distance))
        if exact:
            return exact
        return [s for (sid, d), ss in self._loc_index.items()
                for s in ss if sid == span_id and abs(d - distance) < tol]

    # --- bulk queries ---
    @property
    def sensors(self) -> List[Sensor]:
        return list(self._registry.values())

    @property
    def sensor_ids(self) -> List[str]:
        return list(self._registry.keys())

    @property
    def span_ids(self) -> List[str]:
        return sorted(self._spans)

    @property
    def n_sensors(self) -> int:
        return len(self._registry)

    @property
    def n_spans(self) -> int:
        return len(self._spans)

    def by_axis(self, axis: str) -> List[Sensor]:
        return [s for s in self._registry.values() if s.axis == axis]

    def by_side(self, side: str) -> List[Sensor]:
        return [s for s in self._registry.values() if s.side == side]

    def missing_thresholds(self) -> List[Sensor]:
        return [s for s in self._registry.values() if not s.thresholds.has_any]

    # --- iteration ---
    def __iter__(self) -> Iterator[Span]:
        for sid in sorted(self._spans):
            yield self._spans[sid]

    def __contains__(self, key: Union[str, int]) -> bool:
        if isinstance(key, int):
            return key in self._num_registry
        return key in self._registry

    def __len__(self) -> int:
        return self.n_sensors

    # --- display / export ---
    def summary(self) -> str:
        lines = [f"Bridge: {self.name}", "-" * 65,
                 f"Spans: {self.n_spans}   Sensors: {self.n_sensors}", ""]
        for sp in self:
            pairs = sp.paired()
            lines.append(
                f"  {sp.span_id}  ({len(sp)} sensors, "
                f"{len(sp.unique_distances)} positions, "
                f"{sum(len(v) for v in pairs.values())} paired)")
            for s in sp:
                t = f"{s.thresholds}" if s.thresholds.has_any else "no thresholds"
                num = f"#{s.number:<4d}" if s.number >= 0 else "#?   "
                lines.append(
                    f"    {num} {s.sensor_id:20s}  dist={s.relative_distance:7.2f}  "
                    f"side={s.side or '?':6s}  axis={s.axis or '?':3s}  {t}")
        missing = self.missing_thresholds()
        if missing:
            lines.append(f"\nWARNING: {len(missing)} sensor(s) missing thresholds")
        return "\n".join(lines)

    def to_json(self, path: Union[str, Path], indent: int = 2) -> None:
        data = {
            "name": self.name, "n_spans": self.n_spans, "n_sensors": self.n_sensors,
            "spans": {sid: {"span_id": sid, "sensors": [s.to_dict() for s in sp]}
                      for sid, sp in self._spans.items()},
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=indent, default=str)

    def __repr__(self) -> str:
        return f"Bridge('{self.name}', spans={self.n_spans}, sensors={self.n_sensors})"


# ── Factory function ─────────────────────────────────────────────────────
#
# Separated from the model classes so that:
#   - Bridge/Sensor stay clean domain objects
#   - CSV-specific logic (column names, type conversion) is isolated
#   - A different loader (database, API) can build the same Bridge
#     without changing any model code

def load_bridge(
    position_csv: Union[str, Path],
    threshold_csv: Union[str, Path],
    name: str = "Bridge",
    *,
    delimiter: str = ",",
    pos_cols: Optional[Dict[str, str]] = None,
    thr_cols: Optional[Dict[str, str]] = None,
) -> Bridge:
    """
    Build a Bridge from two CSVs.

    Position CSV  (required: sensor_id, span_id, relative_distance)
                  (optional: side, axis, ... extras captured automatically)

    Threshold CSV (required: sensor_id, threshold_1sigma, threshold_2sigma,
                             threshold_3sigma)
                  (optional: mean_abs, std)

    Thresholds are applied to |value|, so each sigma level is a single
    positive number.  The CSV contains one threshold column per sigma.

    Column names can be remapped via pos_cols / thr_cols dicts for
    different bridges with different CSV formats.
    """
    PC = {
        "sensor_id": "vertical", "span_id": "SECTION",
        "relative_distance": "DIST_M",
        "number": "sensor_id",     # optional: human-friendly sensor number
        "side": "side", "axis": "axis",
    }
    TC = {
        "sensor_id": "sensor",
        "mean_abs": "mean", "std": "std",
        "threshold_1sigma": "threshold_1sigma",
        "threshold_2sigma": "threshold_2sigma",
        "threshold_3sigma": "threshold_3sigma",
    }
    if pos_cols:
        PC.update(pos_cols)
    if thr_cols:
        TC.update(thr_cols)

    bridge = Bridge(name)

    # ---- 1. positions -> sensors ----
    known_keys = set(PC.values())
    rows = _read_csv(position_csv, delimiter)
    created: Dict[str, Sensor] = {}

    for row in rows:
        sid = row[PC["sensor_id"]].strip()

        # Parse number (optional column, defaults to -1)
        num_str = row.get(PC["number"], "").strip()
        num = int(num_str) if num_str.isdigit() else -1

        sensor = Sensor(
            sensor_id=sid,
            span_id=row[PC["span_id"]].strip(),
            relative_distance=_safe_float(row[PC["relative_distance"]]),
            number=num,
            side=row.get(PC["side"], "").strip(),
            axis=row.get(PC["axis"], "").strip(),
            extra={k: v for k, v in row.items() if k not in known_keys},
        )
        bridge.add(sensor)
        created[sid] = sensor

    print(f"[positions]  {len(created)} sensors from {Path(position_csv).name}")

    # ---- 2. thresholds -> attach ----
    thr_rows = _read_csv(threshold_csv, delimiter)
    matched = 0

    for row in thr_rows:
        sid = row[TC["sensor_id"]].strip()
        sensor = created.get(sid)
        if sensor is None:
            continue

        mean_abs = _safe_float(row.get(TC["mean_abs"], "0"))
        std = _safe_float(row.get(TC["std"], "0"))

        for sigma in (1, 2, 3):
            col = TC.get(f"threshold_{sigma}sigma", f"threshold_{sigma}sigma")
            if col in row:
                sensor.thresholds.add(
                    sigma,
                    threshold=_safe_float(row[col]),
                    mean_abs=mean_abs,
                    std=std,
                )
        matched += 1

    print(f"[thresholds] {matched}/{len(thr_rows)} matched from {Path(threshold_csv).name}")
    if matched < len(thr_rows):
        warnings.warn(f"{len(thr_rows)-matched} threshold rows had no matching sensor")

    return bridge


def main():

    parser = argparse.ArgumentParser('get the sensor code by providing the sensor number')

    parser.add_argument('--sensor_num', type = str, help ='provide the sensor number (comma-separated for multiple)')

    position_csv = "/Users/thomas/Desktop/github_repos/industry_time_series/src/dataset/sensors.csv"
    threshold_csv = "/Users/thomas/Desktop/github_repos/industry_time_series/src/dataset/thresholds_abs.csv"
    delimiter: str = ","
    #load_bridge(position_csv, threshold_csv, delimiter=delimiter)


    args = parser.parse_args()
    sensors = args.sensor_num
    sensors = [int(s.strip()) for s in sensors.split(',')]
    #print(sensors)

    Bridge = load_bridge(position_csv, threshold_csv, delimiter=delimiter)
    Bridge.summary()
    #print(Bridge.summary())

    sensors_list =  Bridge.resolve(sensors)
    for i in sensors_list:
        print(i)

if __name__ == "__main__":
    main()

    ###ex: python3 bridge_model.py --sensor_num 106,105,104
    