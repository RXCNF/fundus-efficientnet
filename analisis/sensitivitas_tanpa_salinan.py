"""Pemeriksaan salinan hasil augmentasi (cermin/putaran 180°) dan analisis sensitivitasnya (Reviewer A butir 4d).
Jalur berkas mengikuti susunan folder keluaran notebook dan perlu disesuaikan."""
import pandas as pd, numpy as np, sys, glob, os
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
R='.'
s=pd.read_csv(R+r'\hasil_B_5seed\pembagian_70_15_15.csv'); u=s[s.subset=='uji'].reset_index(drop=True)
u['kunci']=u.file.map(lambda p:'/'.join(p.replace(chr(92),'/').split('/')[-2:]))
C=['cataract','diabetic_retinopathy','glaucoma','normal']; y=u.kelas.map(C.index).values
bocor=set(pd.read_csv(R+r'\hasil\citra_uji_bersalinan.csv').file); m=~u.kunci.isin(bocor).values
print('citra uji:', len(u), '| tanpa salinan:', m.sum())
def load(pat): 
    fs=sorted(glob.glob(pat)); return [np.load(f).argmax(1) for f in fs]
K={'EfficientNetB0':R+r'\hasil_B_5seed\prob_uji_efficientnetb0_none8_s*.npy','+SE r4':R+r'\hasil_B_5seed\prob_uji_efficientnetb0_se4_s*.npy','+SE r8':R+r'\hasil_B_5seed\prob_uji_efficientnetb0_se8_s*.npy','+SE r16':R+r'\hasil_B_5seed\prob_uji_efficientnetb0_se16_s*.npy','+ECA':R+r'\hasil_B_5seed\prob_uji_efficientnetb0_eca8_s*.npy','+CBAM':R+r'\hasil_B_5seed\prob_uji_efficientnetb0_cbam8_s*.npy','ResNet50':R+r'\hasil_B_5seed\prob_uji_resnet50_none8_s*.npy','MobileNetV2':R+r'\hasil_B_5seed\prob_uji_mobilenetv2_none8_s*.npy',
   'Tanpa aug (s0-2)':R+r'\hasil_C_final\prob_uji_efficientnetb0_none8_aug_tanpa_s*.npy','Tanpa aug + SE':R+r'\hasil_C_final\prob_uji_efficientnetb0_se8_aug_tanpa_s*.npy','Tepi':R+r'\hasil_C_final\prob_uji_efficientnetb0_tepi_s*.npy','ViT-S/16':R+r'\hasil_C_final\prob_uji_vit_s16_dasar_s*.npy'}
rows=[]; acc={}
for k,pat in K.items():
    P=load(pat); a=np.array([(p==y).mean() for p in P])*100; b=np.array([(p[m]==y[m]).mean() for p in P])*100
    g=np.array([(p[m&(y==2)]==2).mean() for p in P])*100; gb=np.array([(p[~m&(y==2)]==2).mean() for p in P])*100
    acc[k]=b
    rows.append([k,len(P),f'{a.mean():.2f} ± {a.std(ddof=1):.2f}',f'{b.mean():.2f} ± {b.std(ddof=1):.2f}',f'{(b-a).mean():+.2f}',f'{g.mean():.1f}',f'{gb.mean():.1f}'])
print(pd.DataFrame(rows,columns=['konfigurasi','n','akurasi semua (555)','akurasi tanpa salinan (494)','selisih','sens glaukoma tanpa salinan','sens glaukoma pd citra bersalinan']).to_string())
for k in ['+SE r4','+SE r8','+SE r16','+ECA','+CBAM','ResNet50','MobileNetV2']:
    d=acc[k]-acc['EfficientNetB0']; print(f'{k} vs EfficientNetB0 (tanpa salinan): {d.mean():+.2f} ± {d.std(ddof=1):.2f}, p={stats.ttest_rel(acc[k],acc["EfficientNetB0"]).pvalue:.3f}')
d=acc['Tanpa aug + SE'][:3]-acc['Tanpa aug (s0-2)']; print('SE vs none tanpa aug (s0-2):', d.round(2), 'mean', round(d.mean(),2))
