## 0. GitHub clone (script gomme yerine)

Kod degisikliklerini GitHub'da yapin; Colab her oturumda repoyu klonlar.

### Yerel (PC)

```bash
cd D:\vr-caps\VirtualCapsuleEndoscopy
git init
git add .
git commit -m "VRCaps Colab pipeline"
git remote add origin https://github.com/KULLANICI/g3d.git
git push -u origin main
```

### Colab

1. **Bolum 0** calistirin — `REPO_URL` ve `REPO_BRANCH` ayarlayin
2. Ayni oturumda kodu guncellemek icin Bolum 0'i tekrar calistirin (`git pull`)
3. Notebook'u Drive'a kaydedin; script gomme hucresi artik gerekmez

### Private repo

Colab → Secrets → `GITHUB_TOKEN` (classic PAT, `repo` scope)  
Bolum 0 icindeki yorum satirindaki `userdata.get('GITHUB_TOKEN')` satirini acin.
