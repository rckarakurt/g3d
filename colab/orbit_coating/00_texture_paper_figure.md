## 3c. Makale — stil referansi + UV texture tablosu

Bolum 3 ciktisindan **2 sutunlu tablo** (makale figuru):

| Sol | Sag |
|-----|-----|
| StyleShot stil referansi (`kvasir_style_ref.jpg`) | Uretilen `polyp_uv_texture.png` |

### Cikti

```
/content/paper_texture_figure/
├── paper_texture_ref_table.png
└── texture_validation_report.json
```

Once: **0** → **3** → **3c**

### Texture nasil validate edilir?

1. **Gorsel (qualitative)** — mukoza tonu, damar/doku detayi, duz veya yesil bolge yok mu?
2. **Renk istatistigi** — `lab_delta.deltaE_approx` (referansa yakinlik), `luminance_hist_corr`
3. **Artefakt** — `texture_green_fraction` ≈ 0 olmali
4. **Doku zenginligi** — `texture_entropy_bits` cok dusukse duz/basarisiz uretim
5. **Geometrik tutarlilik** — `ssim_texture_vs_view0`: UV texture ile view_bank 0° render benzerligi
6. **Klinik** — uzman 1–5 skor (makale icin onerilir)

`texture_validation_report.json` icinde tum metrikler ve yorumlar yer alir.
