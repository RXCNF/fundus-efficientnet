"""Pemeriksaan salinan hasil augmentasi (cermin/putaran 180°) dan analisis sensitivitasnya (Reviewer A butir 4d).
Jalur berkas mengikuti susunan folder keluaran notebook dan perlu disesuaikan."""
import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')
r=pd.read_csv(r'hasil\cek_salinan_augmentasi.csv'); r=r[r.selisih_piksel<3].copy()
s=pd.read_csv(r'hasil_B_5seed\pembagian_70_15_15.csv')
s['kunci']=s.file.map(lambda p:'/'.join(p.replace(chr(92),'/').split('/')[-2:]))
sub=dict(zip(s.kunci,s.subset))
r['ka']=r.kelas_a+'/'+r.a; r['kb']=r.kelas_b+'/'+r.b
r['pasang']=[tuple(sorted(t)) for t in zip(r.ka,r.kb)]
u=r.drop_duplicates('pasang').copy()
print('pasangan unik:', len(u), u.transformasi.value_counts().to_dict())
u['sa']=u.ka.map(sub); u['sb']=u.kb.map(sub)
print('salah satu tidak ada di 3.763 citra:', int((u.sa.isna()|u.sb.isna()).sum()))
v=u.dropna(subset=['sa','sb'])
print(pd.crosstab(v.sa, v.sb))
ct=set()
for _,x in v.iterrows():
    if x.sa=='uji' and x.sb in('latih','validasi'): ct.add(x.ka)
    if x.sb=='uji' and x.sa in('latih','validasi'): ct.add(x.kb)
print('citra uji yang punya salinan di latih/validasi:', len(ct), pd.Series([c.split('/')[0] for c in ct]).value_counts().to_dict())
ctl={c for c in ct if any(((x.ka==c and x.sb=='latih') or (x.kb==c and x.sa=='latih')) for _,x in v.iterrows())}
print('  ...yang salinannya di data LATIH:', len(ctl))
pd.Series(sorted(ct)).to_csv(r'hasil\citra_uji_bersalinan.csv', index=False, header=['file'])
