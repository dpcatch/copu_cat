"""
Input helpers.

load_samples: load the global-fit posterior samples (n_sources x n_samples x n_parameters) as a float64 array.

`all_gb_samples.npy` was saved with dtype=object, so a plain `np.load` needs
`allow_pickle=True` and turns every sample into a Python float (roughly 5 GB of RAM for
3590 x 5000 x 8). This module reads the pickle stream directly instead. The stream is only
BINFLOAT opcodes, so it goes straight into a float64 array of about 1.15 GB. A regular
(non-object) .npy is loaded the normal way.
"""
import numpy as np

_BINFLOAT = ord("G")
_MARK, _APPENDS, _MEMOIZE, _FRAME, _TUPLE = ord("("), ord("e"), 0x94, 0x95, ord("t")
_RUN_MAX = 4096  # max consecutive BINFLOAT opcodes decoded per numpy call


def load_samples(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        version = np.lib.format.read_magic(fh)
        shape, fortran_order, dtype = np.lib.format._read_array_header(fh, version)
        header_end = fh.tell()
    if dtype != np.dtype(object):
        return np.load(path)

    buf = np.memmap(path, dtype=np.uint8, mode="r")
    raw = bytes(buf[header_end:header_end + 4096])
    # The object list starts right after EMPTY_LIST (']') + MEMOIZE + MARK.
    start = raw.find(b"]\x94(")
    if start < 0:
        raise ValueError("Unexpected pickle layout: could not find start of object list.")
    p = header_end + start + 2

    n_total = int(np.prod(shape))
    out = np.empty(n_total, dtype=np.float64)
    k = 0
    stride = np.arange(_RUN_MAX) * 9
    while True:
        op = buf[p]
        if op == _BINFLOAT:
            window = buf[p:p + _RUN_MAX * 9]
            n_avail = len(window) // 9
            heads = window[stride[:n_avail]]
            bad = np.flatnonzero(heads != _BINFLOAT)
            n = int(bad[0]) if bad.size else n_avail
            block = np.asarray(window[:n * 9]).reshape(n, 9)[:, 1:]
            out[k:k + n] = np.ascontiguousarray(block).view(">f8").ravel()
            k += n
            p += n * 9
        elif op in (_MARK, _APPENDS, _MEMOIZE):
            p += 1
        elif op == _FRAME:
            p += 9
        elif op == _TUPLE:
            break
        else:
            raise ValueError(f"Unexpected pickle opcode {bytes([op])!r} at byte {p}: "
                             "array contains non-float objects. Use np.load(..., allow_pickle=True).")
    if k != n_total:
        raise ValueError(f"Read {k} values, expected {n_total}.")
    return out.reshape(shape, order="F" if fortran_order else "C")


def source_name(i: int) -> str:
    """Name of recovered source number i (its index along the first axis of the samples)."""
    return f"GF_{i:05d}"


def wrap_angles(df):
    """Polarization has period pi and initial phase has period 2 pi. Map both to [0, period)."""
    df['Polarization'] = np.mod(df['Polarization'], np.pi)
    df['Initial Phase'] = np.mod(df['Initial Phase'], 2 * np.pi)
    return df
