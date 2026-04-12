import struct, sys

path = r"C:\Users\legom\TensorOS\models\google_gemma-4-E2B-it-Q4_0.gguf"
f = open(path, 'rb')

magic = struct.unpack('<I', f.read(4))[0]
version = struct.unpack('<I', f.read(4))[0]
n_tensors = struct.unpack('<Q', f.read(8))[0]
n_kv = struct.unpack('<Q', f.read(8))[0]

# Skip KV pairs
for i in range(n_kv):
    key_len = struct.unpack('<Q', f.read(8))[0]
    key = f.read(key_len).decode('utf-8', errors='replace')
    vtype = struct.unpack('<I', f.read(4))[0]
    if vtype == 8:  # string
        slen = struct.unpack('<Q', f.read(8))[0]
        f.read(slen)
    elif vtype in (5, 6):  # uint32, int32
        f.read(4)
    elif vtype == 10:  # uint64
        f.read(8)
    elif vtype == 7:  # float32
        f.read(4)
    elif vtype in (0, 4):  # uint8, bool
        f.read(1)
    elif vtype == 12:  # uint16
        f.read(2)
    elif vtype == 9:  # array
        atype = struct.unpack('<I', f.read(4))[0]
        acount = struct.unpack('<Q', f.read(8))[0]
        elem_sizes = {0:1, 4:1, 5:4, 6:4, 7:4, 10:8, 12:2}
        if atype == 8:  # string array
            for _ in range(acount):
                slen = struct.unpack('<Q', f.read(8))[0]
                f.read(slen)
        elif atype in elem_sizes:
            f.read(acount * elem_sizes[atype])
        else:
            print(f'Unknown array type {atype}')
            sys.exit(1)
    else:
        print(f'Unknown KV type {vtype}')
        sys.exit(1)

# Read tensor infos
tensors = []
for i in range(n_tensors):
    name_len = struct.unpack('<Q', f.read(8))[0]
    name = f.read(name_len).decode('utf-8')
    n_dims = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
    ttype = struct.unpack('<I', f.read(4))[0]
    offset = struct.unpack('<Q', f.read(8))[0]
    tensors.append((name, dims, ttype, offset))

# Find data start (aligned to 32)
data_start = f.tell()
alignment = 32
if data_start % alignment != 0:
    data_start += alignment - (data_start % alignment)

# Read layer_output_scale values
for name, dims, ttype, offset in tensors:
    if 'layer_output_scale' in name:
        f.seek(data_start + offset)
        val = struct.unpack('<f', f.read(4))[0]
        print(f'{name}: {val:.6f}')

f.close()
