"""Pemeriksaan salinan hasil augmentasi (cermin/putaran 180°) dan analisis sensitivitasnya (Reviewer A butir 4d).
Jalur berkas mengikuti susunan folder keluaran notebook dan perlu disesuaikan."""
import os, numpy as np, pandas as pd, cv2, sys
from scipy.fft import dct
sys.stdout.reconfigure(encoding='utf-8')
D = 'dataset'
files = [(c, os.path.join(D, c, f)) for c in sorted(os.listdir(D)) for f in sorted(os.listdir(os.path.join(D, c)))]
def phash(g):
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    c = dct(dct(g, axis=0, norm='ortho'), axis=1, norm='ortho')[:8, :8]
    return (c > np.median(c)).flatten()
H = {k: [] for k in ['asli', 'flip_h', 'flip_v', 'putar180']}
for c, p in files:
    g = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
    H['asli'].append(phash(g)); H['flip_h'].append(phash(g[:, ::-1])); H['flip_v'].append(phash(g[::-1])); H['putar180'].append(phash(g[::-1, ::-1]))
A = np.array(H['asli'], dtype=np.uint8); kel = np.array([c for c, _ in files])
dist = lambda X, Y: (X[:, None, :] != Y[None, :, :]).sum(2)
d0 = dist(A, A); np.fill_diagonal(d0, 99)
print('pasangan orientasi sama (<=4):', int((np.triu(d0) <= 4).sum()))
hasil = {}
for k in ['flip_h', 'flip_v', 'putar180']:
    T = np.array(H[k], dtype=np.uint8); dk = dist(A, T); np.fill_diagonal(dk, 99)
    i, j = np.where((dk <= 4) & (d0 > 4))    # mirip hanya setelah ditransformasi
    pas = {(min(a, b), max(a, b)) for a, b in zip(i, j)}
    beda = sum(kel[a] != kel[b] for a, b in pas)
    hasil[k] = (len(pas), beda)
    print(f'{k}: {len(pas)} pasangan baru (label berbeda {beda})', [(os.path.basename(files[a][1]), os.path.basename(files[b][1])) for a, b in list(pas)[:4]])

# verifikasi piksel: rata-rata selisih absolut pada 64x64 abu-abu setelah transformasi
def img(p): return cv2.resize(cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
TF = {'flip_h': lambda g: g[:, ::-1], 'flip_v': lambda g: g[::-1], 'putar180': lambda g: g[::-1, ::-1]}
ringkas = []
for k, f in TF.items():
    T = np.array(H[k], dtype=np.uint8); dk = dist(A, T); np.fill_diagonal(dk, 99)
    i, j = np.where((dk <= 4) & (d0 > 4))
    pas = {(min(a, b), max(a, b)) for a, b in zip(i, j)}
    for a, b in pas:
        ga, gb = img(files[a][1]), img(files[b][1])
        mad = min(np.abs(ga - f(gb)).mean(), np.abs(f(ga) - gb).mean())
        ringkas.append((k, os.path.basename(files[a][1]), kel[a], os.path.basename(files[b][1]), kel[b], round(float(mad), 2)))
r = pd.DataFrame(ringkas, columns=['transformasi', 'a', 'kelas_a', 'b', 'kelas_b', 'selisih_piksel'])
iden = r[r.selisih_piksel < 3]
print('\nPasangan yang praktis identik setelah transformasi (selisih < 3):', len(iden), '| label berbeda:', int((iden.kelas_a != iden.kelas_b).sum()))
print(iden.sort_values('selisih_piksel').head(20).to_string())
print('distribusi selisih:', r.selisih_piksel.describe().round(1).to_dict())
r.to_csv(r'hasil\cek_salinan_augmentasi.csv', index=False)
