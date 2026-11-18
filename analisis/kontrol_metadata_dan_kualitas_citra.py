import pandas as pd, numpy as np, sys, os, cv2
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
sys.stdout.reconfigure(encoding='utf-8'); pd.set_option('display.width', 220)
D = 'dataset'
s = pd.read_csv(r'.\hasil_B_5seed\pembagian_70_15_15.csv'); C = sorted(s.kelas.unique())
s['lok'] = s.file.map(lambda p: os.path.join(D, *p.split('/')[-2:])); s['y'] = s.kelas.map({c: i for i, c in enumerate(C)})
s['ekst'] = s.file.str.lower().str.extract(r'(\.\w+)$')[0]
# ---- (1) kontrol metadata: kelas ditebak hanya dari ukuran citra asli dan ekstensi berkas
X = pd.get_dummies(s[['lebar', 'tinggi', 'ekst']].astype(str)).astype(float)
tr, te = s.subset == 'latih', s.subset == 'uji'
for nm, m in [('pohon keputusan', DecisionTreeClassifier(random_state=0)), ('regresi logistik', LogisticRegression(max_iter=2000))]:
    m.fit(X[tr], s.y[tr]); p = m.predict(X[te]); acc = (p == s.y[te]).mean()
    cm = pd.crosstab(s.kelas[te], pd.Series(p, index=s.index[te]).map(dict(enumerate(C))))
    print(f'KONTROL METADATA ({nm}): akurasi uji {acc*100:.2f}% ({(p == s.y[te]).sum()}/{te.sum()})'); print(cm.to_string())
kelompok = s.y.map(lambda v: 0 if C[v] in ('cataract', 'glaucoma') else 1)
m = DecisionTreeClassifier(random_state=0).fit(X[tr], kelompok[tr]); print('dua kelompok sumber (katarak/glaukoma vs RD/normal) dari metadata saja: akurasi uji %.2f%%' % (100 * (m.predict(X[te]) == kelompok[te]).mean()))
# ---- (2) kualitas citra: ketajaman (varians Laplacian), kecerahan, kontras pada 224x224 skala abu-abu
def q(p):
    g = cv2.resize(cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE), (224, 224), interpolation=cv2.INTER_AREA)
    return cv2.Laplacian(g, cv2.CV_64F).var(), g.mean(), g.std()
Q = np.array([q(p) for p in s.lok]); s['tajam'], s['cerah'], s['kontras'] = Q.T
s.drop(columns=['lok']).to_csv(r'.\hasil\kualitas_citra_per_berkas.csv', index=False)
print('\nKUALITAS per kelas (median; seluruh 3.763 citra):'); print(s.groupby('kelas')[['tajam', 'cerah', 'kontras']].median().round(1).to_string())
print('per kelas dan ukuran (median ketajaman):'); print(s.groupby(['kelas', 'lebar']).tajam.agg(['size', 'median']).query('size>=30').round(1).to_string())
u = s[te].reset_index(drop=True)
pa = pd.read_csv(r'.\hasil_A_kaggle\hasil_A\prediksi_uji.csv'); assert (pa.file.map(lambda p: '/'.join(p.split('/')[-2:])) == u.file.map(lambda p: '/'.join(p.split('/')[-2:]))).all()
u['benar_A'] = (pa.label == pa.prediksi).values; u['pred_A'] = pa.prediksi.values
H = r'.\hasil_B_5seed'; u['benar_B'] = np.mean([np.load(f'{H}/prob_uji_efficientnetb0_none8_s{k}.npy').argmax(1) == u.y.values for k in range(5)], 0)
u['tertil'] = pd.qcut(u.tajam, 3, labels=['rendah', 'sedang', 'tinggi'])
print('\nUJI: akurasi menurut tertil ketajaman'); print(u.groupby('tertil', observed=True).agg(n=('y', 'size'), batas_bawah=('tajam', 'min'), batas_atas=('tajam', 'max'), akurasi_model_SE=('benar_A', 'mean'), akurasi_dasar_5seed=('benar_B', 'mean')).round(3).to_string())
print(pd.crosstab(u.kelas, u.tertil).to_string())
print('per kelas x tertil, akurasi model SE:'); print(u.pivot_table(index='kelas', columns='tertil', values='benar_A', aggfunc='mean', observed=True).round(3).to_string())
print('Mann-Whitney ketajaman benar vs salah (model SE): median benar %.1f, salah %.1f, p=%.4f' % (u.tajam[u.benar_A].median(), u.tajam[~u.benar_A].median(), stats.mannwhitneyu(u.tajam[u.benar_A], u.tajam[~u.benar_A]).pvalue))
g = u[u.kelas == 'glaucoma']; print('glaukoma -> katarak:', int((g.pred_A == 'cataract').sum()), '| median ketajaman glaukoma benar %.1f, glaukoma->normal %.1f' % (g.tajam[g.benar_A].median(), g.tajam[g.pred_A == 'normal'].median()))
k = u[u.kelas == 'cataract']; print('katarak: median ketajaman %.1f vs kelas lain %.1f' % (k.tajam.median(), u.tajam[u.kelas != 'cataract'].median()))
low = u[u.tajam < u.tajam.quantile(0.10)]; print('10% citra uji paling buram: n=%d, komposisi %s, diprediksi katarak %d' % (len(low), low.kelas.value_counts().to_dict(), int((low.pred_A == 'cataract').sum())))
ok = u[u.tajam >= u.tajam.quantile(0.10)]; print('akurasi model SE setelah 10%% citra paling buram dibuang: %.2f%% (n=%d); dasar 5 seed: %.2f%%' % (100 * ok.benar_A.mean(), len(ok), 100 * ok.benar_B.mean()))
