"""Kompresör Arıza Tahmini – Gradio arayüzü.

Kayıtlı XGBoost modelini (model/ klasörü) yükler; yüklenen ses kaydını 1 sn'lik
parçalara böler, 40 özellik çıkarır, parçaların oylamasıyla 8 durumdan birini
tahmin eder. Güven kontrolü (Mahalanobis mesafesi) tanımadığı kayıtları işaretler,
SHAP ile kararı en çok etkileyen özellikleri gösterir.

Çalıştırma:  python app/app.py
"""
import os, json
import numpy as np
import pandas as pd
import librosa
import gradio as gr
from scipy.io import wavfile
from scipy.stats import skew, kurtosis
import xgboost as xgb
from xgboost import XGBClassifier

_BURASI = os.path.dirname(os.path.abspath(__file__))
# model/ klasörü uygulamanın yanında (Hugging Face) ya da bir üst klasörde (GitHub deposu) olabilir
KLASOR = next(p for p in [os.path.join(_BURASI, "model"), os.path.join(_BURASI, "..", "model")]
              if os.path.isdir(p))
ORNEK_KLASOR = next((p for p in [os.path.join(_BURASI, "ornekler"), os.path.join(_BURASI, "..", "ornekler")]
                     if os.path.isdir(p)), None)

SINIF_ADLARI = ["Healthy", "Bearing", "Flywheel", "LIV",
                "LOV", "NRV", "Piston", "Riderbelt"]
PARCA_SAYISI, PARCA_UZUNLUGU = 5, 10000
ORNEKLEME_HIZI, N_MFCC = 10000, 13
EGITIM_STD = 0.2246            # eğitim kayıtlarının tipik standart sapması
# En yüksek sınıf olasılığı bu değerin altındaysa sonuç "belirsiz" sayılır.
# Defterin 37. adımında test kümesindeki 4 hatalı tahminin hepsi %70'in altında kalmış,
# bu eşik test kayıtlarının %7,7'sini işaretlemiştir.
BELIRSIZLIK_ESIGI = 0.70

# 1. Model ve özellik sırası
model = XGBClassifier()
model.load_model(os.path.join(KLASOR, "xgb_parca_mfcc_40.json"))
with open(os.path.join(KLASOR, "ozellik_sirasi_40.json"), encoding="utf-8") as f:
    OZELLIK_SIRASI = json.load(f)

# Güven kontrolü (27. adımda kaydedildi; yoksa kontrol yapılmaz)
GUVEN = None
_guven_yolu = os.path.join(KLASOR, "guven_kontrolu.json")
if os.path.exists(_guven_yolu):
    with open(_guven_yolu, encoding="utf-8") as f:
        GUVEN = json.load(f)
    for _k in ["ortalama", "sapma", "merkez", "hassasiyet"]:
        GUVEN[_k] = np.array(GUVEN[_k], dtype=float)
    print(f"Güven kontrolü yüklendi (eşik: {GUVEN['esik']:.2f})")
else:
    print("UYARI: guven_kontrolu.json bulunamadı; güven kontrolü kapalı.")

def benzerlik_mesafesi(X):
    """Eğitim verisine Mahalanobis uzaklığı (parçaların ortalaması)."""
    z = (X[GUVEN["ozellikler"]].values.astype(float) - GUVEN["ortalama"]) / GUVEN["sapma"]
    fark = z - GUVEN["merkez"]
    d2 = np.einsum("ij,jk,ik->i", fark, GUVEN["hassasiyet"], fark)
    return float(np.sqrt(np.maximum(d2, 0)).mean())

# 2. Özellik fonksiyonları (defterdekilerle aynı)
def ozellik_cikar(s):
    s = np.asarray(s, dtype=float).flatten()
    ort = np.mean(s); rms = np.sqrt(np.mean(s ** 2)); tepe = np.max(np.abs(s))
    sp = np.abs(np.fft.rfft(s - ort)); fr = np.fft.rfftfreq(len(s), d=1)
    tg = np.sum(sp)
    return {"Ortalama": ort, "Standart_Sapma": np.std(s), "RMS": rms,
            "Tepe_Deger": tepe, "Crest_Factor": tepe / rms if rms > 0 else 0,
            "Skewness": skew(s), "Kurtosis": kurtosis(s),
            "Spektral_Merkez": np.sum(fr * sp) / tg if tg > 0 else 0,
            "Baskin_Frekans": fr[np.argmax(sp)] if tg > 0 else 0}

def yeni_ozellik_cikar(s):
    s = np.asarray(s, dtype=float); s = s - np.mean(s)
    fr = np.fft.rfftfreq(len(s), d=1.0); g = np.abs(np.fft.rfft(s)) ** 2
    t = np.sum(g)
    if t == 0:
        return dict.fromkeys(["Spektral_Entropi", "Dusuk_Bant_Enerjisi",
                              "Orta_Bant_Enerjisi", "Yuksek_Bant_Enerjisi",
                              "Spectral_Rolloff"], 0.0)
    p = g / t; pp = p[p > 0]
    return {"Spektral_Entropi": -np.sum(pp * np.log2(pp)) / np.log2(len(p)),
            "Dusuk_Bant_Enerjisi": np.sum(g[fr < 0.10]) / t,
            "Orta_Bant_Enerjisi": np.sum(g[(fr >= 0.10) & (fr < 0.25)]) / t,
            "Yuksek_Bant_Enerjisi": np.sum(g[fr >= 0.25]) / t,
            "Spectral_Rolloff": fr[np.searchsorted(np.cumsum(g), 0.85 * t)]}

def mfcc_ozellikleri(parca):
    m = librosa.feature.mfcc(y=np.asarray(parca, dtype=np.float32),
                             sr=ORNEKLEME_HIZI, n_mfcc=N_MFCC,
                             n_fft=512, hop_length=256, n_mels=40)
    oz = {}
    for i in range(N_MFCC):
        oz[f"MFCC_{i+1}_Ort"] = float(m[i].mean())
        oz[f"MFCC_{i+1}_Std"] = float(m[i].std())
    return oz

# 3. Dosya okuma ve tahmin
def dosyadan_sinyal(yol):
    if os.path.splitext(yol)[1].lower() == ".wav":
        _, s = wavfile.read(yol)
        s = np.asarray(s, dtype=float)
        if s.ndim > 1:
            s = s.mean(axis=1)
        if s.std() > 0:
            s = (s - s.mean()) / s.std() * EGITIM_STD
    else:
        s = np.asarray(np.loadtxt(yol, delimiter=","), dtype=float).flatten()
    return s[:PARCA_SAYISI * PARCA_UZUNLUGU]

def tahmin_et(yol):
    s = dosyadan_sinyal(yol)
    satirlar = []
    for k in range(PARCA_SAYISI):
        parca = s[k * PARCA_UZUNLUGU:(k + 1) * PARCA_UZUNLUGU]
        if len(parca) < PARCA_UZUNLUGU:
            break
        satirlar.append({**ozellik_cikar(parca), **yeni_ozellik_cikar(parca),
                         **mfcc_ozellikleri(parca)})
    if not satirlar:
        raise ValueError("Kayıt en az 10.000 örnek (yaklaşık 1 sn) olmalı.")
    X = pd.DataFrame(satirlar)[OZELLIK_SIRASI]
    p = model.predict_proba(X).mean(axis=0)
    sira = np.argsort(p)[::-1]
    mesafe = benzerlik_mesafesi(X) if GUVEN is not None else None
    neden = karar_aciklamasi(X, int(sira[0]))
    return ({SINIF_ADLARI[i]: float(p[i]) for i in sira},
            len(satirlar), mesafe, neden)

def karar_aciklamasi(X, sinif, adet=3):
    """Tahmin edilen sınıf için en etkili özellikler (SHAP, parçaların ortalaması)."""
    katki = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    n, f = X.shape
    if katki.ndim == 2:
        katki = katki.reshape(n, -1, f + 1)
    ort = pd.Series(katki[:, sinif, :f].mean(axis=0), index=X.columns)
    secilen = ort.reindex(ort.abs().sort_values(ascending=False).index).head(adet)
    return ", ".join(f"{ad} ({'+' if v >= 0 else '−'}{abs(v):.2f})".replace(".", ",")
                     for ad, v in secilen.items())

def tr(deger, basamak=1):
    """Sayıyı Türkçe biçimde yazar: ondalık virgül (57,1)."""
    return f"{deger:.{basamak}f}".replace(".", ",")

def arayuz(dosya):
    if dosya is None:
        return "Lütfen bir dosya yükleyin.", None
    yol = dosya if isinstance(dosya, str) else dosya.name
    try:
        olasiliklar, parca_sayisi, mesafe, neden = tahmin_et(yol)
    except Exception as e:
        return f"Dosya okunamadı: {e}", None
    en_iyi = next(iter(olasiliklar))
    durum = "SAĞLIKLI" if en_iyi == "Healthy" else f"ARIZALI – {en_iyi}"
    tahmin_satiri = f"{durum} (%{tr(olasiliklar[en_iyi]*100)})"

    if mesafe is not None and mesafe > GUVEN["esik"]:
        metin = ("⚠️ GÜVENİLİR DEĞİL: Bu kayıt, modelin eğitildiği kompresörün "
                 "seslerine benzemiyor.\n"
                 f"Benzerlik mesafesi {tr(mesafe)} (sınır {tr(GUVEN['esik'], 2)}). "
                 "Farklı bir makine, mikrofon veya ortam olabilir.\n"
                 f"Model yine de şunu söylüyor (yalnızca bilgi amaçlı): {tahmin_satiri}")
    else:
        metin = f"Tahmin: {tahmin_satiri}"
        if olasiliklar[en_iyi] < BELIRSIZLIK_ESIGI:
            ikinci = list(olasiliklar)[1]
            metin += (f"\n⚠️ BELİRSİZ: Model bu karardan yeterince emin değil "
                      f"(güven %{tr(olasiliklar[en_iyi]*100)} < %{BELIRSIZLIK_ESIGI*100:.0f}). "
                      f"İkinci olasılık: {ikinci} (%{tr(olasiliklar[ikinci]*100)}). "
                      "Uzman kontrolü önerilir.")
        if mesafe is not None:
            metin += (f"\n✅ Güven kontrolü: tanıdık kayıt "
                      f"(mesafe {tr(mesafe)} / sınır {tr(GUVEN['esik'], 2)})")
        metin += (f"\n🔎 Kararı en çok etkileyen özellikler: {neden}"
                  "\n   (+ bu sınıfa doğru itti, − bu sınıftan uzaklaştırdı)")
    metin += f"\nKullanılan parça sayısı: {parca_sayisi} / {PARCA_SAYISI}"
    return metin, olasiliklar

# 4. Arayüz
ornekler = sorted(os.path.join(ORNEK_KLASOR, f) for f in os.listdir(ORNEK_KLASOR)) if ORNEK_KLASOR else []

uygulama = gr.Interface(
    fn=arayuz,
    inputs=gr.File(label="Kompresör ses kaydı (.dat / .csv / .txt / .wav)",
                   file_types=[".dat", ".csv", ".txt", ".wav"]),
    outputs=[gr.Textbox(label="Sonuç"),
             gr.Label(label="Sınıf olasılıkları", num_top_classes=8)],
    examples=[[o] for o in ornekler] or None,
    title="Kompresör Arıza Tahmini (Parça + MFCC, XGBoost – Test %98,5)",
    description=("Bir ses kaydı yükleyin; kayıt 1 sn'lik parçalara bölünür ve parçaların oylamasıyla "
                 "8 durumdan biri tahmin edilir. Model yalnızca IIT Kanpur veri setindeki kompresörde "
                 "eğitilmiştir; tanımadığı kayıtlar için 'GÜVENİLİR DEĞİL', emin olmadığı "
                 "tahminler (güven < %70) için 'BELİRSİZ' uyarısı verir. "
                 "Örnekler başka bir kompresöre aittir (Zenodo, CC BY 4.0) ve bu uyarıyı gösterir."),
    flagging_mode="never",
)

if __name__ == "__main__":
    uygulama.launch()
