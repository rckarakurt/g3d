## 0. GitHub clone + temiz baslangic

**Repo:** https://github.com/rckarakurt/g3d

### Sifirdan Colab

1. **Runtime → Restart session**
2. **Bolum 0** calistir (`CLEAN_OLD_OUTPUTS = True` varsayilan)
3. Sirayla: **1 → 1b → 2 → 5b → 3 → 4 → 6 → 7**

### Aci konvansiyonu (view bank)

| Bank acisi | Gorunum |
|------------|---------|
| **0°** | Duz yuz bize bakan (+Z) |
| **-90°** | Sol yan |
| **+90°** | Sag yan |

- Sadece **Y ekseni** (dikey), **-90 .. +90**
- Unity dataset: **Bolum 6** `ensure_geometric_gaze_views()` → `view_bank_az_deg` (geometric + Colab hizali)
- Polyp view bank ayni acilari kullanir (`mesh_rotate_y -az`)

### Kod guncelleme

GitHub push sonrasi Bolum 0 tekrar calistir. Eski notebook kullanma — GitHub'daki `.ipynb` indir.
