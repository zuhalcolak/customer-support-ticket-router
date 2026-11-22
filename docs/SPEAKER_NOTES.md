# Speaker notes – presentation (11 slides)

Simple English (A2–B1). Short sentences. About 30–45 seconds per slide, about 7 minutes in total.
Under each slide there is a short Turkish note (**TR**) so you know what you are saying.
The English text is for speaking. Read it a few times before the presentation.

---

## Slide 1 – Title
> Hello everyone. My name is Zuhal Çolak.
> My project is a Customer Support Ticket Router. This is Use Case 2.
> My system reads support messages in Turkish and English.
> It decides which team should answer the message.
> I compared four different models for this task.

**TR:** Kendini ve projeyi tanıtıyorsun. Mesajları Türkçe/İngilizce okuyup doğru ekibe yönlendiren bir sistem, dört model karşılaştırıldı.

---

## Slide 2 – Problem, users and success
> Imagine a company that sells home appliances, like washing machines and dishwashers.
> Every day, it gets thousands of support messages.
> Some are in Turkish, some are in English.
> A person reads each message and sends it to the right team. This is slow.
> If a message goes to the wrong team, the customer waits longer.
> My users are the support desk and the support agents.
> My goal is simple. The system reads the message, finds the intent, and chooses the department.
> It also finds the product model code, like DW-6534.
> If the system is not sure, it sends the message to a human agent.

**TR:** Problem: çok mesaj var, elle dağıtmak yavaş, yanlış ekip = müşteri bekler. Hedef: niyet → departman, model kodu çıkarma, emin değilse insana gönderme.

---

## Slide 3 – Data
> For data, I used the MASSIVE dataset from Amazon. It has an open license.
> I used the English and the Turkish parts.
> I also wrote 100 example sentences about home appliances myself, in both languages.
> Some messages are "out of scope". This means they are not about our support topics, for example a weather question.
> Some out-of-scope topics are only in the test set. So the models never saw them in training.
> I have 3,808 training, 609 validation and 1,206 test messages.
> I removed duplicate messages and messages that leaked between the splits.
> Before the models, I clean the text. I detect the language, fix Turkish letters, remove links and numbers, and correct spelling mistakes in product names.

**TR:** Veri: MASSIVE (İngilizce+Türkçe) + elle yazılmış 100 cümle. Kapsam dışı konuların bazıları sadece testte var (modeller hiç görmedi). Tekrar eden ve sızan veriler silindi. Ön işleme: dil tespiti, Türkçe harfler, regex temizlik, yazım düzeltme.

---

## Slide 4 – Architecture
> This is the architecture of my system. Every box shows the file in my code.
> On the left, the user types a message in the Streamlit app.
> First, the message goes to preprocessing.
> If the message is a greeting, simple rules answer it. This is like ELIZA.
> If not, four models classify the message.
> Then the confidence logic checks the score.
> If the score is high, the message goes to a department.
> If the score is low, it goes to a human agent.
> At the bottom, you can see the offline training steps.

**TR:** Diyagramı soldan sağa anlat: kullanıcı → ön işleme → (selamsa kural cevabı) → 4 model → güven kontrolü → departman veya insan temsilci. Alt satır: eğitim adımları.

---

## Slide 5 – Technique map
> This slide shows the four technique areas from the course.
> Area A is neural networks. I used FastText vectors and a feed-forward network, trained with SGD.
> Area B is classical NLP. I used regular expressions, edit distance for spelling, ELIZA rules, and TF-IDF with logistic regression.
> Area C is Transformers. I fine-tuned XLM-RoBERTa.
> Area D is efficient models. I used knowledge distillation and INT8 quantization.
> The right column shows where each technique is in my code.

**TR:** Dört teknik alanı tek tek say: A sinir ağı, B klasik NLP, C Transformer, D distilasyon + kuantizasyon. Sağ sütun kodda nerede olduğunu gösteriyor.

---

## Slide 6 – Four models and training
> Here are my four models.
> Model one is TF-IDF with logistic regression. It trains in a few seconds.
> Model two is FastText with a small neural network. I trained it with SGD.
> Model three is XLM-RoBERTa. It is a big multilingual Transformer. It knows Turkish well. Training took about 30 minutes on my laptop GPU.
> Model four is a smaller student model. It learns from XLM-RoBERTa.
> All models use the same labels and the same test set.
> I chose the confidence threshold with the validation set only, not the test set.

**TR:** Dört modeli ve eğitim sürelerini anlat. Hepsi aynı etiketler ve aynı test seti. Eşik sadece validation ile seçildi (teste bakılmadı).

---

## Slide 7 – Distillation and quantization
> This slide is about making the model small and fast.
> First, distillation. A small student model learns from the big teacher model.
> Then, quantization. I changed the weights from 32-bit numbers to 8-bit numbers.
> Look at the table. The big model is 1,112 megabytes. The INT8 model is only 137 megabytes.
> It is about three times faster.
> The accuracy is almost the same: 0.881 and 0.879.
> I also checked that the saved model is really INT8.

**TR:** Distilasyon: küçük öğrenci model büyük öğretmenden öğreniyor. Kuantizasyon: 32 bit → 8 bit. 1112 MB → 137 MB, ~3 kat hızlı, doğruluk neredeyse aynı. Gerçekten INT8 olduğu kontrol edildi.

---

## Slide 8 – Results
> These are the results on the same test set, with 1,206 messages.
> TF-IDF has the best accuracy, 0.895. It is also very small.
> FastText is the fastest, but it has the lowest accuracy.
> XLM-RoBERTa has 0.881 accuracy, but it is very big and slow.
> My INT8 model has 0.879 accuracy. It finds 93 percent of out-of-scope messages.
> I measured speed on the CPU. I ran each model 200 times and took the average.
> The hardware was an Apple M3 laptop.

**TR:** Tabloyu oku: TF-IDF en doğru ve küçük, FastText en hızlı ama en zayıf, XLM-R büyük ve yavaş, INT8 dengeli. Hız CPU'da, 200 ölçüm ortalaması, Apple M3.

---

## Slide 9 – What the comparison showed
> What did I learn?
> The Transformer did not beat the simple model on overall accuracy.
> I think the reason is the data. The messages are short and similar. Simple word matching works well.
> But on my hand-written appliance messages, the Transformer was much better: 0.80 against 0.20.
> These are only 20 sentences, so this result is not very stable.
> The small INT8 model kept the accuracy, but it lost some quality on the appliance messages.

**TR:** Dürüst sonuç: Transformer genel doğrulukta basit modeli geçemedi, çünkü mesajlar kısa ve kalıplı. Ama elle yazılmış zor mesajlarda çok daha iyi (0.80 vs 0.20). Sadece 20 cümle → sonuç oynak.

---

## Slide 10 – Limitations and failure examples
> My system also makes mistakes. Here are some real examples.
> For example, "Bulaşık makinem su boşaltmıyor." My INT8 model thinks it is an installation request. But it is a malfunction. The big model is correct here.
> There are some limitations.
> The data is from virtual assistants, not real support tickets.
> I only have a small number of appliance examples.
> In the future, I want to collect real tickets, add more intents, and make the model even faster.

**TR:** Gerçek hata örneğini göster (bulaşık makinesi → kurulum talebi sanıyor). Sınırlılıklar: veri gerçek destek mesajı değil, beyaz eşya örneği az. Gelecek: gerçek veri, daha fazla niyet, daha hızlı model.

---

## Slide 11 – AI usage
> I used Claude Code, an AI coding assistant.
> It helped me write code, fix bugs, and write documents and these slides.
> It also helped me write the example sentences. I checked all of them.
> But all the numbers in this presentation come from my own scripts. I ran them on my computer.
> No number was copied from an AI answer.
> Thank you for listening. I am happy to answer your questions.

**TR:** Yapay zekâ kullanımını açıkça söyle: kod, hata ayıklama, dokümanlar. Ama tüm sayılar kendi bilgisayarında çalıştırılan scriptlerden geldi. Teşekkür et, soru al.

---

## Possible questions (short answers)

**Why did you choose XLM-RoBERTa?**
> It is multilingual. It knows Turkish and English. It works well with Turkish words and suffixes.

**Why is TF-IDF better than the Transformer?**
> The messages are short and use similar words. Word matching is enough for most of them. The Transformer is better on difficult, new sentences.

**What is out of scope?**
> Messages that are not about our support topics, for example the weather. The system sends them to a human agent.

**How do you choose the threshold?**
> I use the validation set. I choose the highest threshold that loses at most 2 points of accuracy.

**What is quantization?**
> I store the weights as 8-bit numbers instead of 32-bit numbers. The model becomes smaller and faster.

**What is knowledge distillation?**
> A small model learns from a big model. It copies the big model's answers and their probabilities.

**Why is the latency in the app higher than in the table?**
> The app measures one single message, including the tokenizer and the interface. The table shows the average of 200 runs.
