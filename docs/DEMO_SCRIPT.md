# Two-minute demo video – script

**Length:** at most 2:00 (only the first 2:00 is watched). **Format:** screen recording with your
voice. Upload to YouTube (unlisted) or Google Drive ("anyone with the link"), then open the link in
a private browser window to check that it works.

## Before recording (checklist)
1. `source .venv/bin/activate && streamlit run app/main.py` and open http://localhost:8501.
2. Classify one message before you start recording. The first call loads the models into memory.
3. In the **Analytics** tab, press "Reset analytics log" so that the counts start from zero.
4. Open `docs/architecture.png` and `docs/presentation.pdf` (slide 8, results table) in other
   windows or tabs.
5. Zoom the browser to about 90% so the four model cards are visible without scrolling.

Recording tools: macOS **Cmd + Shift + 5**, then "Record Entire Screen" and choose your microphone
under "Options". QuickTime "New Screen Recording" also works.

---

## 0:00 – 0:15 · Problem
**Screen:** the app home page.

**Say (EN):**
> "A home-appliance brand receives thousands of support messages a day in Turkish and English.
> My application reads each message, predicts its intent and routes it to the right team, or to a
> human agent when it is unsure."

**Türkçe açıklama:** Problemi tek cümlede anlatıyorsun. Firmaya Türkçe ve İngilizce binlerce destek
mesajı geliyor. Uygulama her mesajın amacını tahmin edip doğru ekibe gönderiyor, emin olmadığında
insan temsilciye yönlendiriyor.

## 0:15 – 1:15 · Live demo (new input)
**Input 1 – type it yourself, do not click an example** (shows that the app accepts new input):
`My washing machine is not spinning, model WM-8120`

**Say (EN):**
> "I type a new English message. The app detects the language, extracts the model code WM-8120 and
> the product category. The distilled INT8 Transformer, which makes the routing decision, sends it
> to Technical Support with its confidence score. Below, all four models are shown side by side
> with their latency in milliseconds."

**Input 2 – the difficult case, misspelled and informal:**
`my dishwaser wont drain, model dw 6534`

**Say (EN):**
> "Now a difficult case with a spelling mistake. The edit-distance corrector changes 'dishwaser' to
> 'dishwasher', and the model code is still extracted."

**Input 3 – out of scope (Turkish):**
`Yarın İstanbul'da hava nasıl olacak?`

**Say (EN):**
> "A Turkish weather question is out of scope and this topic never appeared in training. The
> model's confidence is only about 20 percent, below the threshold, so the message goes to a human
> agent."

**Input 4 (optional, only if you have time) – greeting:** `Merhaba` → rule-based reply.

**Türkçe açıklama:** En az iki yeni giriş göstermen gerekiyor ve biri zor bir örnek olmalı. Birinci
mesajı kendin yaz, örnek butonuna basma. Ekranda şunları göster: dil rozeti, model kodu, departman,
güven skoru ve dört modelin yan yana kartları (turuncu kart INT8 modeli). Zor örnekte yazım hatası
düzeltmesi "Spelling corrections" satırında görünüyor. Hava durumu sorusunda kırmızı "HUMAN AGENT –
low confidence" kutusu çıkıyor (güven ~%20, eşik 0.43). Videodan önce bu mesajları bir kez dene, çıktıların bu metinle uyuştuğundan
emin ol; uyuşmazsa ekranda gördüğünü söyle.

## 1:15 – 1:45 · Architecture and results
**Screen:** `docs/architecture.png`, then slide 8 (results table) of `docs/presentation.pdf`, or the
"Benchmark results" tab of the app.

**Say (EN):**
> "Here is the architecture. Messages are normalized with regular expressions, model codes are
> extracted and product names are spell-corrected. Then four models run: TF-IDF with logistic
> regression, a FastText feed-forward network trained with SGD, a fine-tuned XLM-RoBERTa, and a
> distilled multilingual DistilBERT quantized to INT8, which runs on the CPU. All four were evaluated
> on the same 1,206-message test set. The INT8 model is about eight times smaller and three times
> faster than XLM-R, with almost the same accuracy: 0.879 against 0.881. Its out-of-scope recall is
> 0.932."

**Türkçe açıklama:** Önce diyagramı göster, kısaca soldan sağa akışı anlat. Sonra sonuç tablosunu göster. Rakamlar
`reports/results.csv` dosyasından geliyor. Tablo: TF-IDF 0.895 / 4.8 MB / 0.74 ms;
FastText 0.780; XLM-R 0.881 / 1112 MB / 18.0 ms; INT8 0.879 / 137 MB / 5.9 ms.

## 1:45 – 2:00 · Conclusion
**Say (EN):**
> "Conclusion: on these short messages the classical baseline is surprisingly strong and slightly
> more accurate overall. The Transformers win on the harder hand-written appliance requests.
> Distillation and quantization made the Transformer small and fast enough for a CPU. Thank you."

**Türkçe açıklama:** Ana sonuç: kısa ve kalıplı mesajlarda klasik model çok güçlü, genel doğrulukta
biraz önde. Transformer modelleri ise elle yazılmış, zor beyaz eşya mesajlarında üstün.
Distilasyon ve kuantizasyon modeli CPU'da çalışacak kadar küçük ve hızlı yaptı. Bu kısım 15
saniye, kısa tut.

---

### Timing tips
* Read the English text out loud once with a timer. If it is over 2:00, skip input 4 first.
* Speak while typing; do not wait in silence while the models run (each takes well under 1 s).
* Do not show any passwords or personal tabs in the recording.
