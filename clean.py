#!/usr/bin/env python3
"""Цэгэн үүлнээс дэвсгэрийг таслах.

COLMAP объект/дэвсгэрийг ялгадаггүй — ширээ, хана бүгд сэргээгддэг. Энэ скрипт:
  1. RANSAC-аар давамгай хавтгайг (ихэвчлэн ширээ) олж хасна
  2. Хавтгайн доод талыг бүхэлд нь хаяна
  3. Воксел дээр холбоост бүлэг ялгаж, хамгийн том бүлгийг (объект) үлдээнэ

Зөвхөн Python-ий стандарт сан ашигладаг — numpy шаардахгүй.

    python3 clean.py вход.ply гарц.ply [--keep-ratio 0.02]
"""
import math, random, struct, sys
from collections import deque

HDR_END = b'end_header\n'
FMT = '<ffffffBBB'          # x y z nx ny nz r g b
REC = struct.calcsize(FMT)  # 27 байт


def read_ply(path):
    with open(path, 'rb') as f:
        raw = f.read()
    i = raw.find(HDR_END)
    if i < 0:
        raise SystemExit('PLY толгой олдсонгүй: ' + path)
    header = raw[:i].decode('ascii', 'replace')
    if 'binary_little_endian' not in header:
        raise SystemExit('Зөвхөн binary_little_endian PLY дэмжинэ.')
    n = 0
    for line in header.splitlines():
        if line.startswith('element vertex'):
            n = int(line.split()[-1])
    body = raw[i + len(HDR_END):]
    need = n * REC
    if len(body) < need:
        raise SystemExit(f'PLY бие дутуу: {len(body)} < {need}')
    return header, n, body[:need]


def write_ply(path, keep_idx, body):
    n = len(keep_idx)
    head = (
        'ply\nformat binary_little_endian 1.0\n'
        f'element vertex {n}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property float nx\nproperty float ny\nproperty float nz\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'end_header\n'
    ).encode('ascii')
    with open(path, 'wb') as f:
        f.write(head)
        out = bytearray(n * REC)
        for j, idx in enumerate(keep_idx):
            out[j * REC:(j + 1) * REC] = body[idx * REC:(idx + 1) * REC]
        f.write(bytes(out))


def main():
    src, dst = sys.argv[1], sys.argv[2]
    keep_ratio_min = 0.02
    if '--keep-ratio' in sys.argv:
        keep_ratio_min = float(sys.argv[sys.argv.index('--keep-ratio') + 1])

    _, n, body = read_ply(src)
    if n == 0:
        raise SystemExit('Цэг алга.')

    # --- координатыг уншина ---
    xs = [0.0] * n; ys = [0.0] * n; zs = [0.0] * n
    for i, rec in enumerate(struct.iter_unpack(FMT, body)):
        xs[i], ys[i], zs[i] = rec[0], rec[1], rec[2]

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    minz, maxz = min(zs), max(zs)
    diag = math.dist((minx, miny, minz), (maxx, maxy, maxz)) or 1.0
    thresh = diag * 0.006          # хавтгайд харьяалагдах зузаан
    print(f'  цэг: {n}  хэмжээ(диагональ): {diag:.3f}  хавтгайн зузаан: {thresh:.4f}')

    # --- RANSAC: давамгай хавтгай ---
    rnd = random.Random(12345)
    sample = rnd.sample(range(n), min(n, 8000))   # оноо тооцох дэд олонлог
    best = (0, None)
    for _ in range(300):
        a, b, c = rnd.sample(range(n), 3)
        ux, uy, uz = xs[b] - xs[a], ys[b] - ys[a], zs[b] - zs[a]
        vx, vy, vz = xs[c] - xs[a], ys[c] - ys[a], zs[c] - zs[a]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        ln = math.sqrt(nx * nx + ny * ny + nz * nz)
        if ln < 1e-12:
            continue
        nx, ny, nz = nx / ln, ny / ln, nz / ln
        d = -(nx * xs[a] + ny * ys[a] + nz * zs[a])
        cnt = 0
        for i in sample:
            if abs(nx * xs[i] + ny * ys[i] + nz * zs[i] + d) < thresh:
                cnt += 1
        if cnt > best[0]:
            best = (cnt, (nx, ny, nz, d))

    if best[1] is None:
        print('  хавтгай олдсонгүй — цэвэрлэлгүй хуулж байна')
        write_ply(dst, list(range(n)), body); return
    nx, ny, nz, d = best[1]
    frac = best[0] / len(sample)
    print(f'  давамгай хавтгай: дэд олонлогийн {frac*100:.1f}% цэгийг эзэлж байна')

    # --- хавтгайг хасах, аль тал нь объект болохыг тодорхойлох ---
    dist = [nx * xs[i] + ny * ys[i] + nz * zs[i] + d for i in range(n)]
    pos = sum(1 for v in dist if v > thresh)
    neg = sum(1 for v in dist if v < -thresh)
    side = 1 if pos >= neg else -1      # цэг олонтой тал = объектын тал
    above = [i for i in range(n) if dist[i] * side > thresh]
    print(f'  хавтгайн дээд тал: {len(above)} цэг ({len(above)/n*100:.1f}%)')
    if len(above) < n * keep_ratio_min:
        print('  хэт цөөн цэг үлдлээ — цэвэрлэлгүй хуулж байна')
        write_ply(dst, list(range(n)), body); return

    # --- воксел дээрх холбоост бүлэг: хамгийн томыг үлдээнэ ---
    vox = diag * 0.012
    grid = {}
    for i in above:
        k = (int(xs[i] / vox), int(ys[i] / vox), int(zs[i] / vox))
        grid.setdefault(k, []).append(i)

    seen, best_comp = set(), []
    neigh = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
             if (dx, dy, dz) != (0, 0, 0)]
    for start in grid:
        if start in seen:
            continue
        comp, q = [], deque([start])
        seen.add(start)
        while q:
            k = q.popleft()
            comp.append(k)
            kx, ky, kz = k
            for dx, dy, dz in neigh:
                nk = (kx + dx, ky + dy, kz + dz)
                if nk in grid and nk not in seen:
                    seen.add(nk); q.append(nk)
        pts = sum(len(grid[k]) for k in comp)
        if pts > sum(len(grid[k]) for k in best_comp) if best_comp else pts > 0:
            best_comp = comp

    keep = [i for k in best_comp for i in grid[k]]
    print(f'  хамгийн том холбоост бүлэг: {len(keep)} цэг ({len(keep)/n*100:.1f}%)')
    if len(keep) < n * keep_ratio_min:
        print('  бүлэг хэт жижиг — хавтгай хассан хувилбарыг үлдээж байна')
        keep = above

    keep.sort()
    write_ply(dst, keep, body)
    print(f'  бичсэн: {dst}  ({len(keep)} цэг)')


if __name__ == '__main__':
    main()
