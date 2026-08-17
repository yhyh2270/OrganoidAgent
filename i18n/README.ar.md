[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Data](https://img.shields.io/badge/Data-Local%20First-4c956c)
![Format](https://img.shields.io/badge/Preview-Multi--format-f4a261)
![Status](https://img.shields.io/badge/Status-Active-success)

`OrganoidAgent` هو إطار عمل خفيف يعتمد على Tornado وواجهة PWA (Progressive Web App) لتصفح ومعاينة مجموعات بيانات الأرغانيود (organoid) محليًا بإعداد بسيط جدًا. يوفر عرضًا مخصصًا للمعاينة حسب نوع الملف للجداول، وصور المجهرية (بما في ذلك TIFF)، والأرشيفات، ونصوص gzip المضغوطة، وكائنات التحليل ` .h5ad` الخاصة بـAnnData.

## 🎯 لمحة سريعة

| الهدف | ما الذي يقدمه هذا المستودع |
|---|---|
| استكشاف بيانات محليًا أولًا | اكتشاف المجموعات، والبيانات الوصفية، وتصفح الملفات من مساحة عمل محلية `datasets/` |
| معاينات غنية | مسارات معاينة للجداول، والصور (بما في ذلك TIFF)، والأرشيفات، و`.gz`، و`.h5ad` |
| واجهة PWA صديقة للعمل دون اتصال | واجهة PWA قابلة للتثبيت مع Service Worker وManifest |
| عمليات عملية | استخراج الأرشيف ومسارات الفهرسة وفقًا للفئات |

## نظرة عامة 🔭

التطبيق الأساسي مصمم لاستكشاف التكوينات التفاعلي للبيانات بقليل من الإعداد:

- واجهة برمجة التطبيقات ومحرك المعاينة في `app.py`
- واجهة PWA في `web/`
- أدوات التنزيل في `scripts/`
- مساحة عمل البيانات المحلية في `datasets/` (مستبعدة من git)

يحتوي هذا المستودع أيضًا على مساحات عمل بحثية/أدوات مجاورة (`BioAgent`، `BioAgentUtils`، `references`، `results`، `vendor`، وسب-وحدة `papers`). زمن التشغيل الأساسي الموصوف في هذا الـREADME هو تطبيق `OrganoidAgent` في المستوى الأعلى.

## المزايا ✨

- فهرسة محلية للمجموعات مع ملخصات الحجم وعدد الملفات
- عرض الملفات بشكل تكراري ضمن المجموعات مع استنتاج نوع الملف
- دعم معاينة لملفات الجداول `CSV/TSV/XLS/XLSX`
- دعم معاينة لصور TIFF/JPG/PNG
- دعم معاينة لملخصات `.h5ad` مع إنشاء معاينة تشتت (scatter) للتضمينات أو PCA
- دعم معاينة لقوائم أرشيفات ZIP/TAR/TGZ + محاولة معاينة أول صورة
- دعم معاينة لأسطر أولى من نص `.gz`
- نقطة نهاية لاستخراج الأرشيفات لمجموعات البيانات المعبأة الكبيرة
- بطاقات بيانات وصفية على مستوى المجموعة تُعرض من Markdown
- واجهة PWA مع Service Worker وManifest
- تنقية أساسية لمسارات الملفات (`safe_dataset_path`) لحصر الوصول إلى الملفات داخل `datasets/`

### لمحة سريعة

| المجال | ما الذي يوفّره |
|---|---|
| اكتشاف المجموعات | قوائم datasets على مستوى المجلد مع عدد الملفات وملخصات الحجم |
| استكشاف الملفات | عرض تكراري واستنتاج نوع الملف (`image`، `table`، `analysis`، `archive`، وغير ذلك) |
| معاينات غنية | الجداول، صور TIFF/الصور، مقتطفات نص gzip، محتويات الأرشيفات، ملخصات AnnData |
| مرئيات التحليل | معاينات نقطية `.h5ad` من تضمينات `obsm` أو بديل PCA |
| دعم الحزم | عرض أرشيف + نقطة استخراج للحزم المضغوطة الكبيرة |
| تجربة الويب | PWA قابلة للتثبيت مع أصول Service Worker المهيأة للعمل دون اتصال |

## بنية المشروع 🗂️

```text
OrganoidAgent/
├─ app.py
├─ web/
│  ├─ index.html
│  ├─ app.js
│  ├─ styles.css
│  ├─ sw.js
│  ├─ manifest.json
│  └─ icons/
├─ scripts/
│  ├─ download_organoid_datasets.py
│  ├─ download_drug_screening_datasets.py
│  └─ overlay_segmentations.py
├─ datasets/                      # downloaded data and preview cache (git-ignored)
├─ metadata/
│  └─ zenodo_10643410.md
├─ papers/                        # submodule: prompt-is-all-you-need
├─ i18n/                          # currently present for multilingual README files
├─ BioAgent/                      # related but separate app
├─ BioAgentUtils/                 # related training/data utilities
├─ references/
├─ results/
└─ vendor/                        # external submodules (copilot-sdk, paper-agent, codex)
```

## المتطلبات ✅

- Python `3.10+`
- مدير البيئة الموصى به: `conda` أو `venv`

الحزم المطلوبة/الاختيارية المستنتجة من الكود:

| الحزمة | الدور |
|---|---|
| `tornado` | مطلوب لبدء تشغيل السيرفر |
| `pandas` | اختياري: دعم معاينة الجداول |
| `anndata`, `numpy` | اختياري: معاينة `.h5ad` ورسم تحليلي |
| `Pillow` | اختياري: عرض الصور وإنشاء معاينات |
| `tifffile` | اختياري: دعم معاينة TIFF |
| `requests` | اختياري: سكربتات تنزيل البيانات |
| `kaggle` | اختياري: تنزيلات Kaggle في سكربت فحص العقاقير |

ملاحظة افتراضية: لا يوجد حاليًا `requirements.txt` أو `pyproject.toml` أو `environment.yml` للمشروع الرئيسي في الجذر.

## التثبيت ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# الخيار A: conda (مثال)
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# الخيار B: التشغيل الأساسي فقط
pip install tornado
```

## الاستخدام 🚀

### البداية السريعة

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # اختياري إذا كانت المتطلبات مثبتة مسبقًا
python app.py --port 8080
```

افتح `http://localhost:8080`.

### اختبار سريع لـ API

```bash
curl http://localhost:8080/api/datasets
```

### تنزيل البيانات (اختياري)

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

يتم تخزين البيانات المنزلة في `datasets/` (مستبعدة من git).

## نقاط النهاية API 🌐

| الطريقة | المسار | الغرض |
|---|---|---|
| `GET` | `/api/datasets` | عرض قوائم المجموعات مع إحصائيات موجزة |
| `GET` | `/api/datasets/{name}` | عرض ملفات مجموعة واحدة |
| `GET` | `/api/datasets/{name}/metadata` | إرجاع بطاقة وصفية بصيغة markdown |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | قوائم ملفات موجهة حسب الفئة |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | حمولة معاينة مدركة لنوع الملف |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | استخراج الأرشيف في مجلد `_extracted` المماثل |
| `GET` | `/files/<path>` | خدمة الملفات الخام ضمن datasets |
| `GET` | `/previews/<path>` | خدمة أصول معاينة مُنشأة |

مثال استدعاء معاينة:

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## الإعدادات 🧩

الإعدادات الحالية للتشغيل مصممة لتكون مبسطة عمدًا:

- منفذ الخادم: وسيط `--port` في `app.py` (الافتراضي `8080`)
- دليل البيانات: ثابت على `datasets/` بالنسبة إلى جذر المستودع
- ذاكرة التخزين المؤقت للمعاينات: `datasets/.cache/previews`
- ربط البيانات الوصفية: قاموس `DATASET_METADATA` في `app.py`
- رمز GitHub API للتحميل (اختياري): متغير البيئة `GITHUB_TOKEN` أو الوسيطة `--github-token`

ملاحظة افتراضية: إذا كنت تحتاج مسارات datasets قابلة للتخصيص أو إعدادات خادم الإنتاج، فهذه الخيارات غير مكشوفة بعد في ملفات إعدادات المستوى الأعلى.

## الأمثلة 🧪

### تصفح الملفات وفق فئات محددة

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### استخراج أرشيف

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### تشغيل أوضاع التنزيل الانتقائي

```bash
# Organoid datasets: skip GEO, keep Zenodo
python scripts/download_organoid_datasets.py --skip-geo

# Drug-screening datasets: only Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## ملاحظات التطوير 🛠️

- الخادم الخلفي يخدم أصول الواجهة الأمامية الثابتة من `web/`.
- Service Worker وManifest موجودان في `web/sw.js` و`web/manifest.json`.
- التوجيه حسب نوع الملف والمعاينات مطبَّقة في `app.py`.
- التحقق اليدوي (إرشادات المشروع الحالية): الواجهة PWA تفتح على `http://localhost:8080`
- التحقق اليدوي (إرشادات المشروع الحالية): `/api/datasets` تُرجع JSON
- التحقق اليدوي (إرشادات المشروع الحالية): المعاينات تُعرض لملفات CSV/XLSX/الصور/الأرشيفات

## استكشاف الأخطاء 🩺

- `ModuleNotFoundError` لمكتبات المعاينة: ثبّت الحزم الناقصة (`pandas`, `anndata`, `numpy`, `Pillow`, `tifffile`).
- قائمة بيانات فارغة: تأكد من وجود البيانات تحت `datasets/` وأن المجلدات ليست ذات أسماء تبدأ بنقطة.
- معاينة `.h5ad` لا تظهر صورة التشتت: تحقق من تثبيت `anndata` و`numpy` و`Pillow`.
- مشاكل معاينة/استخراج أرشيف كبير: استخدم نقطة استخراج وحلل الملفات المستخرجة مباشرة.
- أخطاء حد معدل GitHub downloader: زوّد `GITHUB_TOKEN` عبر متغير البيئة أو خيار CLI.
- تنزيل Kaggle لا يعمل: ثبّت `kaggle` واضبط بيانات الاعتماد في `~/.kaggle/kaggle.json`.

## خارطة الطريق 🧭

تحسينات مرجحة مقبلة (غير منفذة بالكامل بعد في تطبيق الجذر):

- إضافة ملف تبعية على مستوى الجذر (`requirements.txt` أو `pyproject.toml`)
- إضافة اختبارات آلية لمعالجات API ودوال المعاينة
- إضافة إعداد مرن لجذر datasets وإعدادات cache
- إضافة ملف تشغيل إنتاجي واضح (غير debug، مع إرشادات reverse-proxy)
- توسيع التوثيق متعدد اللغات تحت `i18n/`

## المساهمة 🤝

المساهمات مرحب بها. سير عمل عملي:

1. أنشئ fork وقم بإنشاء فرعًا مركّزًا.
2. أبقِ التغييرات مقتصرة على منطقة منطقية واحدة.
3. تحقق يدويًا من تشغيل التطبيق والنقاط الرئيسية.
4. افتح PR مع ملخص، والأوامر المستخدمة، ولقطات شاشة لتغييرات الواجهة.

اتفاقيات نمط الكود المحلية في هذا المستودع:

- Python: إزاحة 4 مسافات، snake_case للدوال/الملفات، CapWords للفئات
- أبقِ منطق الواجهة الأمامية في `web/app.js` لهذا التطبيق (تجنب إعادة كتابة أطر العمل غير الضرورية)
- اجعل التعليقات موجزة وتقتصر على المواضع غير الواضحة منطقًا

## ملخص المشروع (رسمي) 📌

- `app.py`: سيرفر Tornado ومسارات API.
- `web/`: أصول PWA.
- `scripts/`: أدوات مساعدة لتنزيل البيانات.
- `datasets/`: تخزين البيانات المحلية.
- `papers/`: وحدة فرعية تحتوي مواد المراجع.

## الترخيص 📄

لا يوجد ملف `LICENSE` في جذر المشروع حاليًا.

ملاحظة افتراضية: إلى أن تُضاف رخصة على مستوى الجذر، يُرجى اعتبار شروط إعادة الاستخدام/إعادة التوزيع غير محددة لمشروع `OrganoidAgent` على مستوى أعلى.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
