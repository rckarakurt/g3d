## 3c. Makale — stil referansi + UV texture (ECCV)

Bolum 3 ciktisindan **ECCV/LNCS** formatinda 2 panelli figur:

| Sol (a) | Sag (b) |
|---------|---------|
| Style exemplar | Generated UV texture |

### Cikti (`/content/paper_texture_figure/`)

- `paper_texture_ref_table.png` — 300 DPI, 122 mm genislik
- `paper_texture_ref_table.pdf` — vektor sarim (matplotlib varsa)
- `texture_validation_table.tex` — ECCV `booktabs` tablosu
- `texture_validation_report.json` — metrikler + `latex_caption_hint`

Once: **0** → **3** → **3c**

Preamble:
```latex
\usepackage{booktabs}
```

Figur caption ornegi `texture_validation_report.json` icindeki `latex_caption_hint` alaninda.
