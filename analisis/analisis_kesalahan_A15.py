"""Analisis kesalahan tambahan (Reviewer A butir 8b dan 15).

Masukan (semuanya ada di repositori / keluaran notebook):
  hasil_B_5seed/prob_uji_<konfig>_s<seed>.npy   probabilitas data uji 40 run (8 konfigurasi x 5 seed)
  hasil_B_5seed/pembagian_70_15_15.csv           pembagian data
  hasil_A/prediksi_uji.csv                        prediksi model EfficientNetB0+SE (Bagian A)
  hasil/kualitas_citra_per_berkas.csv             ketajaman (varians Laplacian), kecerahan, kontras per citra

Keluaran: hasil/analisis_kesalahan_A15.json dan hasil/konsistensi_kesalahan_per_citra.csv
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'
B = os.path.join(ROOT, 'hasil_B_5seed'); OUT = os.path.join(ROOT, 'hasil')
C = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
KONFIG = ['efficientnetb0_none8', 'efficientnetb0_se4', 'efficientnetb0_se8', 'efficientnetb0_se16',
          'efficientnetb0_eca8', 'efficientnetb0_cbam8', 'resnet50_none8', 'mobilenetv2_none8']
key = lambda p: '/'.join(str(p).replace('\\', '/').split('/')[-2:])

q = pd.read_csv(os.path.join(OUT, 'kualitas_citra_per_berkas.csv'))
u = q[q.subset == 'uji'].reset_index(drop=True)          # urutan sama dengan berkas prob_uji (diperiksa di bawah)
y = u.kelas.map(C.index).values
pa = pd.read_csv(os.path.join(ROOT, r'hasil_A_kaggle\hasil_A\prediksi_uji.csv'))
assert (pa.file.map(key) == u.file.map(key)).all()
pA = pa[[f'p_{c}' for c in C]].values; predA = pA.argmax(1)
P = {k: np.stack([np.load(os.path.join(B, f'prob_uji_{k}_s{s}.npy')) for s in range(5)]) for k in KONFIG}
for k in KONFIG: assert P[k].shape == (5, len(u), 4)
R = {}

# ---------- (1) konsistensi kesalahan lintas seed
salah = {k: P[k].argmax(2) != y for k in KONFIG}                  # (5, n)
def jaccard(a, b): return (a & b).sum() / max(1, (a | b).sum())
R['jaccard_kesalahan_antarseed'] = {k: round(float(np.mean([jaccard(salah[k][i], salah[k][j]) for i, j in itertools.combinations(range(5), 2)])), 3) for k in KONFIG}
semua = np.concatenate([salah[k] for k in KONFIG])                  # (40, n)
n_salah = semua.sum(0)
u['jumlah_run_salah_dari_40'] = n_salah
u['jumlah_seed_salah_dasar'] = salah['efficientnetb0_none8'].sum(0)
u['jumlah_seed_salah_se8'] = salah['efficientnetb0_se8'].sum(0)
R['citra_selalu_benar_40_run'] = int((n_salah == 0).sum())
R['citra_salah_di_semua_40_run'] = int((n_salah == 40).sum())
R['citra_salah_di_>=32_run (80%)'] = int((n_salah >= 32).sum())
keras = u[n_salah >= 32]
R['citra_sulit_per_kelas'] = keras.kelas.value_counts().to_dict()
# prediksi mayoritas untuk citra sulit
maj = np.array([np.bincount(np.concatenate([P[k].argmax(2)[:, i] for k in KONFIG]), minlength=4).argmax() for i in range(len(u))])
R['citra_sulit_kelas_asli_ke_prediksi_mayoritas'] = pd.crosstab(keras.kelas, pd.Series(maj[keras.index], index=keras.index).map(dict(enumerate(C)))).to_dict()
d = u.jumlah_seed_salah_dasar
R['dasar_5seed_distribusi_jumlah_seed_salah'] = {int(i): int((d == i).sum()) for i in range(6)}
R['dasar_5seed_kesalahan_konsisten_(5/5)_dari_kesalahan_seed0'] = f"{int(((d == 5)).sum())} dari {int(salah['efficientnetb0_none8'][0].sum())}"
R['model_SE_A_kesalahan_yang_juga_salah_di_>=4_seed_dasar'] = f"{int(((predA != y) & (d >= 4)).sum())} dari {int((predA != y).sum())}"
R['kualitas_citra_sulit_vs_lain_median_ketajaman'] = [round(float(keras.tajam.median()), 1), round(float(u.tajam[n_salah < 32].median()), 1),
                                                     float(stats.mannwhitneyu(keras.tajam, u.tajam[n_salah < 32]).pvalue)]

# ---------- (2) glaukoma <-> katarak dan kualitas citra
med_kelas = u.groupby('kelas')[['tajam', 'cerah', 'kontras']].median()
def rinci(mask, nama):
    rows = u[mask][['file', 'kelas', 'lebar', 'tinggi', 'tajam', 'cerah', 'kontras', 'jumlah_run_salah_dari_40']].copy()
    rows['file'] = rows.file.map(key)
    rows['persentil_tajam_dlm_kelas_asli'] = [round(float(stats.percentileofscore(u.tajam[u.kelas == k], t)), 0) for k, t in zip(rows.kelas, rows.tajam)]
    return rows.round(1).to_dict('records')
R['model_SE_glaukoma_ke_katarak'] = rinci((y == 2) & (predA == 0), 'g2c')
R['model_SE_katarak_ke_glaukoma'] = rinci((y == 0) & (predA == 2), 'c2g')
R['median_kualitas_per_kelas'] = med_kelas.round(1).to_dict()
# pada 40 run: rata-rata jumlah kesalahan G->C dan C->G per run, dan ketajaman citra yang terlibat
g2c = np.concatenate([(P[k].argmax(2) == 0) & (y == 2) for k in KONFIG]); c2g = np.concatenate([(P[k].argmax(2) == 2) & (y == 0) for k in KONFIG])
R['40run_rata2_glaukoma_ke_katarak_per_run'] = round(float(g2c.sum(1).mean()), 2)
R['40run_rata2_katarak_ke_glaukoma_per_run'] = round(float(c2g.sum(1).mean()), 2)
ig = g2c.any(0); ic = c2g.any(0)
R['40run_ketajaman_citra_G_ke_C_median_vs_glaukoma_lain'] = [round(float(u.tajam[ig].median()), 1), round(float(u.tajam[(y == 2) & ~ig].median()), 1), int(ig.sum())]
R['40run_ketajaman_citra_C_ke_G_median_vs_katarak_lain'] = [round(float(u.tajam[ic].median()), 1), round(float(u.tajam[(y == 0) & ~ic].median()), 1), int(ic.sum())]

# ---------- (3) tingkat keyakinan pada prediksi salah
def keyakinan(prob, yy):
    conf = prob.max(1); ok = prob.argmax(1) == yy
    return {'median_keyakinan_benar': round(float(np.median(conf[ok])), 3), 'median_keyakinan_salah': round(float(np.median(conf[~ok])), 3),
            'salah_dengan_keyakinan_>=0,9': f'{int((~ok & (conf >= 0.9)).sum())} dari {int((~ok).sum())}',
            'salah_dengan_keyakinan_<0,6': f'{int((~ok & (conf < 0.6)).sum())} dari {int((~ok).sum())}',
            'AUROC_keyakinan_membedakan_benar_salah': round(float(roc_auc_score(ok, conf)), 3),
            'akurasi_jika_keyakinan_<0,6_dirujuk': [round(float(ok[conf >= 0.6].mean()), 4), f'dirujuk {int((conf < 0.6).sum())} citra ({100 * (conf < 0.6).mean():.1f}%)']}
R['keyakinan_model_SE_bagianA'] = keyakinan(pA, y)
R['keyakinan_dasar_5seed_(gabungan)'] = keyakinan(P['efficientnetb0_none8'].reshape(-1, 4), np.tile(y, 5))

# ---------- (4) 10% citra uji paling buram
t10 = u.tajam.quantile(0.10); low = u.tajam < t10
benar_dasar = (~salah['efficientnetb0_none8']).mean(0)
R['10pct_paling_buram'] = {'batas_ketajaman': round(float(t10), 1), 'n': int(low.sum()), 'komposisi': u.kelas[low].value_counts().to_dict(),
                           'diprediksi_katarak_oleh_model_SE': int((predA[low] == 0).sum()),
                           'akurasi_model_SE_pada_10pct_buram': round(float((predA == y)[low].mean()), 4),
                           'akurasi_model_SE_tanpa_10pct_buram': round(float((predA == y)[~low].mean()), 4),
                           'akurasi_dasar_5seed_pada_10pct_buram': round(float(benar_dasar[low].mean()), 4),
                           'akurasi_dasar_5seed_tanpa_10pct_buram': round(float(benar_dasar[~low].mean()), 4)}

u.assign(file=u.file.map(key))[['file', 'kelas', 'tajam', 'jumlah_run_salah_dari_40', 'jumlah_seed_salah_dasar', 'jumlah_seed_salah_se8']].to_csv(
    os.path.join(OUT, 'konsistensi_kesalahan_per_citra.csv'), index=False)
json.dump(R, open(os.path.join(OUT, 'analisis_kesalahan_A15.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
