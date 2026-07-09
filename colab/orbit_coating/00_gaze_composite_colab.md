## 7. Aci eslestirmeli composite (Unity + mesh)

Her Unity karesine, **aynı bakis acisindaki** StyleShot mesh render yapilir.

### Mantik

```
Unity kare N
  view_bank_az_deg = +42.3°   (gaze_views.csv, geometric + Colab hizali)
       ↓
view_bank: view_az040.png + view_az045.png  (blend)
       ↓
ekran merkezine (anchor) yapistir → composite
```

### Gerekli

| Kaynak | Yol |
|--------|-----|
| Unity RGB + gaze | `/content/medical_gan_dataset/` |
| Mesh view bank | `/content/ply_styleshot_out/view_bank/` |
| gaze_views.csv | Bolum 6 (`ensure_geometric_gaze_views`) |

### Cikti

```
/content/gaze_composite/
├── rgb/000000.png ...      ← polipli Unity kareleri
├── debug/match_*.png       ← Unity | mesh | composite (aci etiketli)
├── trajectory_composite.mp4
└── composite_manifest.csv  ← frame, view_bank_az_deg, bank_azimuth
```

### Calistirma

Once: **5b** (Drive dataset) → **6** (geometric gaze export) → **3** (view bank, Unity acilariyla) → **7**

Bolum 6: `unity_dataset_angles.ensure_geometric_gaze_views()` — `view_bank_az_deg` uretir.

Bolum 7 ayarlari:
- `STRICT_ANGLE_MATCH = True` — kare bazli aci eslestirme
- `WRITE_DEBUG = True` — aci kontrolu icin uc lu onizleme
