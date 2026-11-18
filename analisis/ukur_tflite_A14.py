import os, sys, json, time, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np, pandas as pd, psutil
import tensorflow as tf
from tensorflow import keras
L = keras.layers
P = psutil.Process()
mode = sys.argv[1]
ROOT = '.'
OUT = os.path.join(ROOT, r'hasil\a14_tflite'); os.makedirs(OUT, exist_ok=True)
DS = 'dataset'
W = os.path.join(ROOT, r'hasil_A_kaggle\hasil_A\model_terbaik.weights.h5')
IMG = (224, 224); CLS = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
pred = pd.read_csv(os.path.join(ROOT, r'hasil_A_kaggle\hasil_A\prediksi_uji.csv'))
files = [os.path.join(DS, *f.split('/dataset/')[1].split('/')) for f in pred.file]
y = np.array([CLS.index(l) for l in pred.label])
rss = lambda: P.memory_info().rss / 2**20

def load_img(p):
    img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
    return tf.image.resize(img, IMG, method='bilinear').numpy().astype(np.float32)

class RandomCropResize(L.Layer):
    def __init__(self, crop=200, out=224, seed=None, **kw):
        super().__init__(**kw); self.crop, self.out, self.seed = crop, out, seed
    def call(self, x, training=None):
        return x

import h5py, re as _re
def _sorted(keys, cls):
    ks = [k for k in keys if _re.fullmatch(cls + r'(_\d+)?', k)]
    return sorted(ks, key=lambda k: int(k.rsplit('_', 1)[1]) if _re.search(r'_\d+$', k) and k != cls else 0)
def _vars(g): return [g['vars'][str(i)][()] for i in range(len(g['vars']))]
def build():
    base = keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=IMG + (3,))
    f = h5py.File(W, 'r'); gb = f['layers/functional/layers']; keys = list(gb.keys())
    CLSMAP = {'Conv2D': 'conv2d', 'BatchNormalization': 'batch_normalization', 'DepthwiseConv2D': 'depthwise_conv2d', 'Normalization': 'normalization'}
    cnt = {}
    for lyr in base.layers:
        if not lyr.weights: continue
        c = CLSMAP[type(lyr).__name__]; i = cnt.get(c, 0); cnt[c] = i + 1
        k = _sorted(keys, c)[i]; v = _vars(gb[k])
        assert [w.shape for w in lyr.get_weights()] == [x.shape for x in v], (lyr.name, k)
        if c == 'normalization':   # versi Kaggle (weights='imagenet') membagi lagi dengan sqrt(std); setara dengan variance = std^2
            v = [v[0], v[1] ** 2, v[2]]
        lyr.set_weights(v)
    inp = L.Input(IMG + (3,), name='citra')
    x = base(inp, training=False)
    C = x.shape[-1]
    s = L.GlobalAveragePooling2D()(x); s = L.Reshape((1, 1, C))(s)
    d1 = L.Dense(C // 8, activation='relu'); d2 = L.Dense(C, activation='sigmoid')
    s = d2(d1(s)); x = L.Multiply()([x, s])
    x = L.GlobalAveragePooling2D()(x); b1 = L.BatchNormalization(); x = b1(x)
    d3 = L.Dense(128, activation='relu'); x = d3(x); b2 = L.BatchNormalization(); x = b2(x)
    x = L.Dropout(0.3)(x); d4 = L.Dense(4, activation='softmax'); out = d4(x)
    for lyr, k in [(d1, 'dense'), (d2, 'dense_1'), (d3, 'dense_2'), (d4, 'dense_3'), (b1, 'batch_normalization'), (b2, 'batch_normalization_1')]:
        lyr.set_weights(_vars(f['layers'][k]))
    return keras.Model(inp, out)

if mode == 'keras':
    r0 = rss(); m = build(); r1 = rss()
    X = np.stack([load_img(f) for f in files])
    p = m.predict(X, batch_size=32, verbose=0)
    acc = float((p.argmax(1) == y).mean())
    print('akurasi keras', acc, 'cocok dengan prediksi Kaggle:', float((p.argmax(1) == pred.prediksi.map(CLS.index).values).mean()),
          'selisih prob maks', float(np.abs(p - pred[[f'p_{c}' for c in CLS]].values).max()))
    np.save(os.path.join(OUT, 'X_uji.npy'), X)
    # konversi TFLite
    sizes = {}
    for nama, opt in [('fp32', None), ('fp16', 'fp16'), ('int8_dinamis', 'dyn')]:
        cv = tf.lite.TFLiteConverter.from_keras_model(m)
        if opt: cv.optimizations = [tf.lite.Optimize.DEFAULT]
        if opt == 'fp16': cv.target_spec.supported_types = [tf.float16]
        b = cv.convert(); fn = os.path.join(OUT, f'model_{nama}.tflite'); open(fn, 'wb').write(b)
        sizes[nama] = round(len(b) / 2**20, 2)
    json.dump({'akurasi_keras_lokal': acc, 'ukuran_tflite_MB': sizes}, open(os.path.join(OUT, 'keras.json'), 'w'), indent=1)
    print(sizes)

elif mode == 'kerasmem':   # memori dan latensi Keras CPU, batch 1
    tf.config.threading.set_inter_op_parallelism_threads(1)
    r0 = rss(); m = build(); X = np.load(os.path.join(OUT, 'X_uji.npy'))[:60]
    f = tf.function(lambda x: m(x, training=False))
    for i in range(5): f(X[i:i+1])
    t = []
    pk = rss()
    for i in range(5, 55):
        a = time.perf_counter(); f(X[i:i+1]).numpy(); t.append(time.perf_counter() - a); pk = max(pk, rss())
    res = {'rss_awal_MB': r0, 'rss_puncak_MB': pk, 'tambahan_MB': pk - r0, 'latensi_ms_median': 1000 * float(np.median(t))}
    print(res); json.dump(res, open(os.path.join(OUT, 'kerasmem.json'), 'w'), indent=1)

elif mode == 'tflite':
    nama, thr = sys.argv[2], int(sys.argv[3]); full_eval = len(sys.argv) > 4
    X = np.load(os.path.join(OUT, 'X_uji.npy'))
    r0 = rss()
    it = tf.lite.Interpreter(model_path=os.path.join(OUT, f'model_{nama}.tflite'), num_threads=thr)
    it.allocate_tensors(); ii = it.get_input_details()[0]['index']; oi = it.get_output_details()[0]['index']
    def run(x):
        it.set_tensor(ii, x[None]); it.invoke(); return it.get_tensor(oi)[0]
    for i in range(5): run(X[i])
    t = []; pk = rss()
    for i in range(5, 105):
        a = time.perf_counter(); run(X[i]); t.append(time.perf_counter() - a); pk = max(pk, rss())
    res = {'model': nama, 'thread': thr, 'rss_awal_MB': r0, 'rss_puncak_MB': pk, 'tambahan_MB': pk - r0,
           'latensi_ms_median': 1000 * float(np.median(t)), 'latensi_ms_p90': 1000 * float(np.percentile(t, 90))}
    if full_eval:
        p = np.stack([run(x) for x in X])
        res['akurasi_uji'] = float((p.argmax(1) == y).mean()); res['n_benar'] = int((p.argmax(1) == y).sum())
        ref = pred[[f'p_{c}' for c in CLS]].values
        res['prediksi_sama_dgn_kaggle'] = float((p.argmax(1) == ref.argmax(1)).mean())
    print(res); json.dump(res, open(os.path.join(OUT, f'tflite_{nama}_{thr}.json'), 'w'), indent=1)
