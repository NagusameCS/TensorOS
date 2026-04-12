import struct, mmap
path = 'models/google_gemma-4-E2B-it-Q4_0.gguf'
f = open(path, 'rb')
mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data = mm[:]

offset = 0
offset += 4+4+8+8  # skip header

type_sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
n_kv = struct.unpack_from('<Q', data, 16)[0]

for i in range(n_kv):
    key_len = struct.unpack_from('<Q', data, offset)[0]; offset += 8
    key = data[offset:offset+key_len].decode('utf-8', errors='replace'); offset += key_len
    vtype = struct.unpack_from('<I', data, offset)[0]; offset += 4
    
    if vtype == 8:
        slen = struct.unpack_from('<Q', data, offset)[0]; offset += 8
        val = data[offset:offset+slen].decode('utf-8', errors='replace'); offset += slen
        if 'rope' in key or 'attention' in key or 'head' in key:
            print(f'{key} = "{val}"')
    elif vtype == 9:
        atype = struct.unpack_from('<I', data, offset)[0]; offset += 4
        acnt = struct.unpack_from('<Q', data, offset)[0]; offset += 8
        if atype == 8:
            for _ in range(acnt):
                slen2 = struct.unpack_from('<Q', data, offset)[0]; offset += 8; offset += slen2
        elif atype in type_sizes:
            if 'rope' in key or 'attention' in key or 'head' in key:
                print(f'{key} = array[{acnt}] of type {atype}')
            offset += acnt * type_sizes[atype]
    elif vtype in type_sizes:
        sz = type_sizes[vtype]
        if vtype == 6:
            val = struct.unpack_from('<f', data, offset)[0]
        elif vtype == 4:
            val = struct.unpack_from('<I', data, offset)[0]
        elif vtype == 5:
            val = struct.unpack_from('<i', data, offset)[0]
        elif vtype == 7:
            val = struct.unpack_from('<B', data, offset)[0]
        elif vtype == 10:
            val = struct.unpack_from('<Q', data, offset)[0]
        else:
            val = '?'
        if 'rope' in key or 'attention' in key or 'head' in key:
            print(f'{key} ({vtype}) = {val}')
        offset += sz
    else:
        print(f'Unknown vtype {vtype}'); break
        
mm.close(); f.close()
