## 5. Unity dataset — Drive yolu

### Drive (kaynak)

```
MyDrive/vrcaps/medical_gan_dataset/
├── camera.json
├── rgb/
├── depth/
└── poses/
```

Colab tam yol:
```
/content/drive/MyDrive/vrcaps/medical_gan_dataset
```

### Calisma kopyasi (Colab oturumu)

Bolum 5b bu klasoru **`/content/medical_gan_dataset/`** altina kopyalar.

### Bolum 5b

```python
SOURCE = "drive_folder"   # varsayilan
```

1. `drive.mount()`
2. `vrcaps/medical_gan_dataset` klasorunu kopyala
3. Klasor yoksa `vrcaps/unity-vr-caps-dataset.rar` dener

### Diger /content yollari

| Cikti | Yol |
|-------|-----|
| Polyp render | `/content/ply_styleshot_out/view_bank/` |
| Composite | `/content/gaze_composite/` |
