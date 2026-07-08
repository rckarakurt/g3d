## 0. GitHub clone (script gomme yerine)

Kod degisikliklerini GitHub'da yapin; Colab her oturumda repoyu klonlar.

### Yerel (PC)

```bash
cd D:\vr-caps\VirtualCapsuleEndoscopy
git init
git add .
git commit -m "VRCaps Colab pipeline"
git remote add origin https://github.com/rckarakurt/g3d.git
git push -u origin main
```

### Colab

1. **Bolum 0** calistirin — repo: `https://github.com/rckarakurt/g3d.git`
2. Kodu guncellemek icin Bolum 0'i tekrar calistirin (`git pull`)
