## 7. Aci eslestirmeli composite (Unity + Colab polyp)

Her Unity karesine, **aynı view-bank acisindaki** Colab polyp render yapilir.

### Mantik

```
Unity kare N
  view_bank_az_deg = +32.1°   (geometric + Colab hizali, gaze_views.csv)
       ↓
view_bank: view_azm030.png + view_azp035.png  (blend)
       ↓
ekran merkezine (anchor) yapistir → composite
```

### Gerekli

| Kaynak | Yol |
|--------|-----|
| Unity RGB + depth + poses | `/content/medical_gan_dataset/` |
| Colab polyp view bank | `/content/ply_styleshot_out/view_bank/` |
| gaze_views.csv | Bolum 6 veya Bolum 7 otomatik (`ensure_geometric_gaze_views`) |

### Cikti

```
/content/gaze_composite/
├── rgb/000000.png ...      ← polipli Unity kareleri
├── debug/match_*.png       ← Unity | mesh | composite (aci etiketli)
├── trajectory_composite.mp4
└── composite_manifest.csv  ← frame, view_bank_az_deg, bank_azimuth
```

### Calistirma

Once: **5b** (Drive dataset) → **3** (view bank) → **6** (gaze acilari, opsiyonel) → **7**

Bolum 7 ayarlari:
- `STRICT_ANGLE_MATCH = True` — kare bazli `view_bank_az_deg` eslestirme
- `WRITE_DEBUG = True` — aci kontrolu icin uc lu onizleme
