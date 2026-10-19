"""Minimal PSD reader: layer list + layer pixels + merged image."""
import struct, sys, zlib
import numpy as np

def u8(b, o): return b[o], o + 1
def u16(b, o): return struct.unpack_from('>H', b, o)[0], o + 2
def i16(b, o): return struct.unpack_from('>h', b, o)[0], o + 2
def u32(b, o): return struct.unpack_from('>I', b, o)[0], o + 4
def i32(b, o): return struct.unpack_from('>i', b, o)[0], o + 4


def unpack_bits_rows(data, o, rows, cols, counts):
    out = np.zeros((rows, cols), np.uint8)
    for r in range(rows):
        n = counts[r]
        seg = data[o:o + n]
        o += n
        row = bytearray()
        i = 0
        while i < len(seg) and len(row) < cols:
            c = seg[i]; i += 1
            if c < 128:
                row += seg[i:i + c + 1]; i += c + 1
            elif c > 128:
                row += bytes([seg[i]]) * (257 - c); i += 1
        row = row[:cols]
        out[r, :len(row)] = np.frombuffer(bytes(row), np.uint8)
    return out, o


class Layer:
    pass


def read_desc_keys(b):
    return b


def parse(path, want_pixels=True):
    b = open(path, 'rb').read()
    assert b[:4] == b'8BPS'
    ver, = struct.unpack_from('>H', b, 4)
    ch, h, w, depth, mode = struct.unpack_from('>HIIHH', b, 12)
    o = 26
    n, o = u32(b, o); o += n          # color mode
    n, o = u32(b, o); res_start = o; o += n  # image resources
    n, o = u32(b, o) if ver == 1 else (struct.unpack_from('>Q', b, o)[0], o + 8)
    lm_end = o + n
    li_len, o = u32(b, o) if ver == 1 else (struct.unpack_from('>Q', b, o)[0], o + 8)
    li_end = o + li_len
    cnt, o = i16(b, o)
    cnt = abs(cnt)
    layers = []
    for _ in range(cnt):
        L = Layer()
        L.top, L.left, L.bottom, L.right = struct.unpack_from('>iiii', b, o); o += 16
        nch, o = u16(b, o)
        L.channels = []
        for _ in range(nch):
            cid, o = i16(b, o)
            ln, o = u32(b, o) if ver == 1 else (struct.unpack_from('>Q', b, o)[0], o + 8)
            L.channels.append((cid, ln))
        o += 4
        L.mode = b[o:o + 4].decode('latin1'); o += 4
        L.opacity = b[o]; L.clip = b[o + 1]; L.flags = b[o + 2]; o += 4
        exlen, o = u32(b, o)
        ex_end = o + exlen
        mlen, o = u32(b, o)
        L.mask = None
        if mlen:
            L.mask = struct.unpack_from('>iiii', b, o) + (b[o + 16], b[o + 17])
        o += mlen
        blen, o = u32(b, o); L.blendif = b[o:o + blen]; o += blen
        nl = b[o]; L.name = b[o + 1:o + 1 + nl].decode('latin1')
        o += ((nl + 1 + 3) // 4) * 4
        L.info = {}
        while o < ex_end:
            sig = b[o:o + 4]; key = b[o + 4:o + 8].decode('latin1'); o += 8
            big = ver == 2 and key in ('LMsk', 'Lr16', 'Lr32', 'Layr', 'Mt16', 'Mt32', 'Mtrn', 'Alph', 'FMsk', 'lnk2', 'FEid', 'FXid', 'PxSD')
            if big:
                ln = struct.unpack_from('>Q', b, o)[0]; o += 8
            else:
                ln, o = u32(b, o)
            L.info[key] = b[o:o + ln]
            o += ln
            if ln % 2 and False:
                o += 1
        if 'luni' in L.info:
            d = L.info['luni']; n2 = struct.unpack_from('>I', d, 0)[0]
            L.name = d[4:4 + 2 * n2].decode('utf-16-be').rstrip('\x00')
        o = ex_end
        layers.append(L)
    # channel image data
    for L in layers:
        L.pix = {}
        hh, ww = L.bottom - L.top, L.right - L.left
        for cid, ln in L.channels:
            start = o
            comp, = struct.unpack_from('>H', b, o)
            if cid == -2 and L.mask:
                t, l, bt, r = L.mask[:4]; rh, rw = bt - t, r - l
            else:
                rh, rw = hh, ww
            if want_pixels and rh > 0 and rw > 0 and ln > 2:
                if comp == 0:
                    L.pix[cid] = np.frombuffer(b, np.uint8, rh * rw, o + 2).reshape(rh, rw)
                elif comp == 1:
                    if ver == 1:
                        counts = np.frombuffer(b, '>u2', rh, o + 2)
                        d0 = o + 2 + 2 * rh
                    else:
                        counts = np.frombuffer(b, '>u4', rh, o + 2)
                        d0 = o + 2 + 4 * rh
                    L.pix[cid], _ = unpack_bits_rows(b, d0, rh, rw, counts.astype(int))
                elif comp in (2, 3):
                    raw = zlib.decompress(b[o + 2:o + ln])
                    a = np.frombuffer(raw, np.uint8)[:rh * rw].reshape(rh, rw)
                    if comp == 3:
                        a = np.cumsum(a.astype(np.uint16), axis=1).astype(np.uint8)
                    L.pix[cid] = a
            o = start + ln
    return dict(w=w, h=h, ch=ch, mode=mode, depth=depth, ver=ver, layers=layers,
                buf=b, lm_end=lm_end, res=(res_start))


def merged(info):
    b = info['buf']; o = info['lm_end']
    comp, = struct.unpack_from('>H', b, o); o += 2
    h, w, ch = info['h'], info['w'], info['ch']
    if comp == 1:
        nrows = h * ch
        cs = 2 if info['ver'] == 1 else 4
        counts = np.frombuffer(b, '>u2' if cs == 2 else '>u4', nrows, o).astype(int)
        o += cs * nrows
        planes = []
        for c in range(ch):
            p, o = unpack_bits_rows(b, o, h, w, counts[c * h:(c + 1) * h])
            planes.append(p)
    else:
        planes = [np.frombuffer(b, np.uint8, h * w, o + c * h * w).reshape(h, w) for c in range(ch)]
    return np.dstack(planes[:3])


def rgba(info, L):
    hh, ww = L.bottom - L.top, L.right - L.left
    if hh <= 0 or ww <= 0:
        return None
    z = np.zeros((hh, ww), np.uint8)
    r, g, bl = L.pix.get(0, z), L.pix.get(1, z), L.pix.get(2, z)
    a = L.pix.get(-1, np.full((hh, ww), 255, np.uint8))
    return np.dstack([r, g, bl, a])


if __name__ == '__main__':
    info = parse(sys.argv[1], want_pixels=False)
    print(info['w'], info['h'], 'ch', info['ch'], 'mode', info['mode'], 'depth', info['depth'], 'ver', info['ver'])
    depth = 0
    for i, L in enumerate(info['layers']):
        sec = L.info.get('lsct')
        st = struct.unpack_from('>I', sec, 0)[0] if sec else None
        vis = 'hidden' if L.flags & 2 else ''
        print(i, repr(L.name), L.mode, L.opacity, 'clip' if L.clip else '', vis,
              (L.top, L.left, L.bottom, L.right), 'sect', st, sorted(L.info.keys()),
              'mask' if L.mask else '')
