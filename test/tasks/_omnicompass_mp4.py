"""Read-only MP4 table parser for synthetic offset-fault test fixtures."""

import struct


def require(ok, message):
    if not ok:
        raise ValueError(message)


def atoms(handle, start, end):
    position = start
    while position < end:
        handle.seek(position)
        head = handle.read(8)
        require(len(head) == 8, "Truncated atom")
        size, kind = struct.unpack(">I4s", head)
        header = 8
        if size == 1:
            size = struct.unpack(">Q", handle.read(8))[0]
            header = 16
        elif size == 0:
            size = end - position
        require(size >= header and position + size <= end, "Invalid atom boundary")
        yield kind, position + header, position + size
        position += size


def children(handle, parent):
    return list(atoms(handle, parent[1], parent[2]))


def only(entries, kind):
    found = [entry for entry in entries if entry[0] == kind]
    require(len(found) == 1, f"Expected one {kind!r}")
    return found[0]


def video_offsets(handle, size):
    moov = only(list(atoms(handle, 0, size)), b"moov")
    tables = []
    for trak in (a for a in children(handle, moov) if a[0] == b"trak"):
        mdia = only(children(handle, trak), b"mdia")
        hdlr = only(children(handle, mdia), b"hdlr")
        handle.seek(hdlr[1] + 8)
        if handle.read(4) != b"vide":
            continue
        minf = only(children(handle, mdia), b"minf")
        stbl = only(children(handle, minf), b"stbl")
        for kind, start, end in children(handle, stbl):
            if kind not in (b"stco", b"co64"):
                continue
            handle.seek(start + 4)
            count = struct.unpack(">I", handle.read(4))[0]
            width = 4 if kind == b"stco" else 8
            require(start + 8 + count * width == end, "Offset table size mismatch")
            table = []
            for index in range(count):
                position = start + 8 + index * width
                value = int.from_bytes(handle.read(width), "big")
                table.append((value, position, width))
            tables.append(table)
    require(len(tables) == 1, "Only one video chunk-offset table is supported")
    return tables[0]
