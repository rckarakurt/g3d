## 8. Makale — 7 sabit aci sentetik composite

Unity mucosa + StyleShot textured polyp (view bank), **ortaya yapıştırılmış**.

### Sabit eslestirme

| Unity frame | Bank acisi | Polyp view bank |
|-------------|------------|-----------------|
| image_0067 | -45° | view_azm045.png |
| image_0078 | -30° | view_azm030.png |
| image_0087 | -15° | view_azm015.png |
| image_0098 | 0° | view_azp000.png |
| image_0121 | +15° | view_azp015.png |
| image_0132 | +30° | view_azp030.png |
| image_0143 | +45° | view_azp045.png |

### Gerekli

- Bolum **3** — `/content/ply_styleshot_out/view_bank/`
- Bolum **5b** — `/content/medical_gan_dataset/`

### Cikti

```
/content/paper_angle_composites/
├── singles/composite_f0067_azm045.png ...  (7 ayri PNG)
├── pairs_stacked/pair_composite_f0067_azm045.png  (Unity|Polyp|Composite ust uste)
├── paper_pairs_strip_7.png        (7 kolon, her kolon ust uste)
├── paper_composites_strip_7.png   (sadece composite yan yana)
├── paper_composites_stack_7.png   (7 composite ust uste tek sutun)
└── paper_composites_manifest.json
```

Once: **0** (git pull) → **5b** → **3** → **8**
