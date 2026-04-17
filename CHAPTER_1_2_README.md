# Chapter 1: Introduction

Phishing is one of the most effective social engineering methods that hackers use to get personal information, credentials, and gain access to an organization. Phishing communications are sent over email, text, QR codes, or phone calls, typically trying to get victims to open a link or attachment that leads to a credential harvesting form or malware. Spam is becoming the most widespread type of cybercrime, with 3.4 billion emails sent every day (Ronish et al., 2025). According to the FBI, phishing is the most common Internet crime, with $54 million lost in 2020.

Phishing emails are a major problem for businesses worldwide. They enter inboxes disguised as emails from well-known brands, using realistic themes to manipulate recipients. The problem is worsening as businesses migrate to the cloud (Dutta, 2024), and as AI tools enable attackers to craft high-quality, grammatically flawless phishing emails that are increasingly difficult to distinguish from legitimate ones (AI and Fraud, 2024).

Modern email services have added malicious email filters and authentication methods to detect suspicious senders, yet phishing emails continue to be missed and misclassified (Takashi et al., 2024). Many systems flag emails as potentially fraudulent without explaining why, leaving recipients unable to make informed judgments. To combat this at scale, organizations need effective phishing detection techniques — and systems leveraging LLM embeddings can significantly bolster these efforts (Basit et al., 2021; Sarker, 2023).

---

## 1.1 Problem Background

Numerous studies have applied machine learning to phishing email classification, deriving phishing-specific characteristics from email headers and bodies for binary (ham/spam) categorization (Beaman et al., 2022; Mughaid et al., 2022; Nabeel et al., 2021; Sonowal et al., 2020). NLP techniques have also been used to analyze email content and discern context and intent (Che et al., 2017; Li et al., 2022), ranging from frequency-based approaches like TF-IDF and Bag-of-Words to deep learning models like BERT.

Modern ML techniques categorize features from sender data, URLs, content, headers, and stylometric patterns (Jovanovic et al., 2024; Ilias et al., 2024; Salloum et al., 2022). However, these methods mostly provide statistical detection results without clear explanations, struggle with diverse multilingual datasets, and suffer from overfitting, poor adaptability, and the need for frequent retraining (Kyaw et al., 2024).

Large Language Models (LLMs) are advanced AI systems trained on vast amounts of text to understand and generate natural language. Models like GPT-4 excel at detecting intricate language patterns and anomalies (Hua et al., 2024; Qin et al., 2024), and their ability to comprehend context, semantics, and subtle linguistic signals enables them to detect phishing emails that evade standard filters (Jiang et al., 2024). However, LLMs are often considered "black boxes," with decision-making processes that are difficult to explain — a key challenge for classification tasks (Luo & Specia, 2024).

---

## 1.2 Research Aim and Objectives

This study aims to design and assess a phishing email detection system that employs LLM embeddings to improve the accuracy and reliability of phishing email identification, with a focus on semantic comprehension of email content to enhance user awareness.

### Objectives

- To assess existing literature on the application of LLMs in cybersecurity, particularly for email threat analysis and phishing identification.
- To assemble and categorize a labeled dataset of authentic and phishing emails from publicly available sources for model training and assessment.
- To compare and analyze LLM embedding techniques — including BERT, RoBERTa, and DistilBERT — to identify the most effective method for representing email text.
- To train a machine learning classifier using the selected embedding technique and assess its ability to distinguish phishing from legitimate emails.
- To evaluate the proposed detection system using standard metrics: precision, recall, F1-score, and accuracy.
- To assess the practical applicability of the proposed LLM-based method relative to conventional techniques.

---

## 1.3 Research Questions

- How effective are LLM embeddings in identifying and classifying phishing emails compared to other machine learning techniques?
- Which LLM embedding technique — BERT, RoBERTa, or DistilBERT — best represents email content for phishing detection?
- How can LLM-based phishing detection systems be developed to raise user awareness and enhance response mechanisms?
- How effective are LLM-based phishing detection models against new and zero-day phishing techniques?
- What are the most important linguistic and semantic characteristics of LLM embeddings that indicate phishing, and what is the trade-off between embedding complexity and classification accuracy?

---

# Chapter 2: Literature Review

## 2.1 Depth and Relevance to the Problem

Secure email communication is the foundation of digital transactions across business, government, and healthcare. Poor email security — including weak authentication and filtering — can escalate into data breaches, financial fraud, and reputational damage (Lastdrager, 2014; Khonji et al., 2013). High-risk sectors such as banking, healthcare, and government are especially attractive targets for advanced persistent threats and social engineering attacks that evade conventional controls (Hong, 2012; Jakobsson & Myers, 2006).

Small and medium-sized businesses are particularly vulnerable, as they often lack dedicated security teams and large budgets. Without strong email filters or security training, phishing attempts frequently succeed, leading to costly incident responses (Parsons et al., 2014; Arachchilage & Love, 2014). Studying past phishing incidents can help researchers identify protection priorities and training needs (Vishwanath et al., 2011; Kumaraguru et al., 2010).

Poor phishing detection — including improper filtering and slow responses — leads to employees divulging credentials, installing malware, and compromising systems (Alkhalil et al., 2021). Effective countermeasures must account for usability and false positive rates, and must evolve to counter threats like spear phishing and clone phishing (Abu-Nimeh et al., 2007).

---

### 2.1.1 Traditional Methods for Phishing Detection

Rule-based filtering and blacklisting have historically been the primary defenses against phishing. These systems use predefined rules based on sender reputation, keywords, URLs, and attachments (Khonji et al., 2013; Fette et al., 2007). However, they are rigid, require manual updates, cannot detect zero-day phishing, and are easily evaded by skilled attackers (Basit et al., 2021; Chiew et al., 2018).

Heuristic and feature-based ML approaches — using Naive Bayes, Decision Trees, SVM, and Random Forests — improved upon rule-based systems but still rely on manually extracted features, which are susceptible to feature drift and do not capture semantic meaning. High false positive rates and inability to handle sophisticated, realistic phishing emails remain key limitations (Gupta et al., 2018; Heartfield & Loukas, 2016).

---

### 2.1.2 Machine Learning and Deep Learning Paradigms in Cybersecurity

Machine learning has become a viable paradigm for automating security functions like threat detection, malware classification, intrusion detection, and anomaly detection (Apruzzese et al., 2018; Buczak & Guven, 2016). Key methodologies include:

- **Supervised Learning:** Trains classifiers on labeled data to distinguish phishing from legitimate emails, using algorithms such as SVM, Random Forest, CNNs, RNNs, and Transformers (Apruzzese et al., 2018).
- **Unsupervised Learning:** Identifies anomalies from unlabeled data using clustering (k-means, DBSCAN), PCA, and Autoencoders (Buczak & Guven, 2016).
- **Semi-Supervised & Reinforcement Learning:** Combines small labeled datasets with large unlabeled ones, or trains adaptive security agents through interaction with simulated environments (Xin et al., 2018).

Deep learning architectures are particularly effective for unstructured data: CNNs for spatial feature extraction, RNNs/LSTMs for sequential data, and Transformer models for long-range semantic understanding in text (Devlin et al., 2019; Brown et al., 2020). Challenges include the need for large datasets, adversarial vulnerability, interpretability, and high computational cost.

---

### 2.1.3 Natural Language Processing and Text Classification for Phishing Detection

NLP is central to phishing detection given the text-based nature of the attack vector. Early approaches used handcrafted lexical, syntactic, and semantic features with classifiers like Bag-of-Words, TF-IDF, and n-grams (Khonji et al., 2013; Gupta et al., 2018). While effective to a degree, these struggled with linguistic nuance.

Word embeddings such as Word2Vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014) improved text classification by representing words as semantic vectors, enabling models to recognize relationships between terms like "urgent" and "immediate" or "password" and "credentials." However, traditional word embeddings lack contextual awareness — the word "bank" carries the same vector regardless of meaning.

Contextual word embeddings, as used in modern LLMs, resolve this by encoding meaning relative to surrounding words — a critical capability for detecting phishing, where attackers exploit subtle, context-dependent language (Ariyadasa et al., 2023).

---

### 2.1.4 Large Language Models and Transformer Architecture

LLMs are built on the Transformer architecture (Vaswani et al., 2017), which uses a self-attention mechanism to weigh the importance of words relative to each other in a sequence — significantly outperforming older RNN and LSTM architectures. The encoder component of the Transformer produces contextualized embeddings used for classification tasks.

Bidirectional processing, as demonstrated in BERT, enables a deeper understanding of language by reading text from both directions simultaneously (Devlin et al., 2019). Pre-training on large corpora followed by fine-tuning on task-specific data allows LLMs to perform well even with limited labeled examples — an important advantage for phishing detection (Devlin et al., 2019; Liu et al., 2019).

---

### 2.1.5 BERT: Architecture, Capabilities, and Applications

BERT (Devlin et al., 2019) introduced bidirectional contextualized language modeling through two pre-training objectives: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).

**Strengths:** Deep bidirectional context understanding; state-of-the-art NLP performance; excellent transfer learning with minimal task-specific training; available in Base and Large variants.

**Weaknesses:** High resource consumption, especially the Large variant; NSP objective not always beneficial; large model size limits flexibility.

---

### 2.1.6 RoBERTa: Optimized BERT Training

RoBERTa (Liu et al., 2019) improves on BERT by removing the NSP objective, using dynamic masking, larger batch sizes, and significantly more training data — resulting in consistently higher downstream task accuracy.

**Strengths:** Higher accuracy than BERT on most benchmarks; greater training stability; more robust representations; retains BERT's strong transfer learning properties.

**Weaknesses:** Inherits BERT's computational intensity; longer training time; gains over BERT are marginal on some tasks.

---

### 2.1.7 DistilBERT: Efficient Knowledge Distillation

DistilBERT (Sanh et al., 2019) uses knowledge distillation to train a smaller "student" model to mimic the original BERT "teacher." With only 6 layers instead of 12, it is 40% smaller and 60% faster at inference, while retaining 97% of BERT's language understanding.

**Strengths:** Fast inference; lower memory and compute requirements; competitive accuracy on most tasks; reduced environmental impact.

**Weaknesses:** Slightly lower accuracy than BERT and RoBERTa; may struggle with tasks requiring deep semantic understanding or subtle linguistic nuance.

---

### 2.1.8 Application of LLM Embeddings in Phishing Detection

Recent studies show significant improvements in phishing detection using LLM embeddings. Ariyadasa et al. (2023) found that BERT embeddings outperform TF-IDF-based approaches, particularly for sophisticated phishing emails that avoid obvious trigger words — achieving over 95% accuracy on benchmark datasets. The key advantage of LLMs is their ability to capture underlying semantic meaning, which traditional models cannot.

---

## 2.2 Quality of the Sources

The rapid advancement of LLMs has created both new capabilities and new vulnerabilities in phishing detection. Koide et al. (2024) demonstrated GPT-4's ability to detect phishing and explain its decisions, while Heiding et al. (2025) showed the same models can be used to craft highly convincing phishing emails — a paradox where identical technology serves both offense and defense.

Deep learning research has explored CNN-BiGRU models, bidirectional LSTM with novel encoding schemes, and transformer-based comparisons of TF-IDF, Word2Vec, and BERT — though most studies evaluate clean data rather than adversarial conditions.

LLMs' instruction-following capabilities also introduce risks: Greshake et al. (2023) demonstrated indirect prompt injection attacks on LLM-integrated applications, Liu et al. (2023) showed that malicious input can override system prompts, and Liu et al. (2024) confirmed that current LLMs cannot reliably differentiate legitimate from malicious input.

In multilingual environments, research has shown reduced phishing detection accuracy in low-resource languages, and studies have explored dynamic learning strategies and evaluations of open- and closed-source LLMs with varied prompt strategies.

---

## 2.3 Critical Evaluation

Despite extensive research on phishing email detection, key gaps remain. Most researchers have used spam datasets rather than phishing-specific datasets — a meaningful distinction, as not all spam constitutes phishing. NLP-based systems relying on ML tend to process surface-level word patterns without true semantic understanding, making them vulnerable to slight linguistic variations or synonym substitutions (Salloum et al., 2022).

Evaluation frameworks are fragmented: Tusher et al. (2025) explored zero-shot classification via category mapping; Xuan et al. (2025) used multilingual benchmarks for advanced LLM evaluation; Chinta et al. (2025) proposed performance measures for high- and low-resource languages. These evaluate individual techniques but do not address the overall interaction of vulnerabilities.

Deployment of LLM-based phishing detectors also faces scalability challenges — BERT and RoBERTa demand significant compute, making real-time filtering of millions of emails costly. Efforts to improve efficiency include DistilBERT, quantization, and pruning (Sanh et al., 2019). Adversarial attacks — crafting emails specifically designed to evade LLM detectors — are an ongoing concern (Biggio & Roli, 2018).

Explainability is a critical factor for adoption. Security analysts need to understand why an email was flagged, and researchers are working on attention visualization and SHAP values to provide interpretable explanations for LLM classification decisions (Ribeiro et al., 2016; Lundberg & Lee, 2017).

---

### 2.3.1 Human Factors in Phishing Defense and User Awareness

Despite advances in technology, people remain the weakest link in cybersecurity. Phishing exploits cognitive biases, emotional appeals, and trust signals (Vishwanath et al., 2011). Parsons et al. (2014) found significant variation in employee security awareness, with most lacking sufficient knowledge to defend against spear phishing.

Security awareness training is a critical complement to automated detection. Kumaraguru et al. (2010) demonstrated that embedded training with immediate feedback is far more effective than conventional lecture-style approaches. An LLM-based detection system that provides explanations alongside phishing alerts could serve dual purposes — automated protection and user education.

Vishwanath et al. (2011) identified three factors that make people fall for phishing: heuristic processing (gut instinct), habituation (instinctive responses to email patterns), and trust cues (visual and contextual signals). Time pressure, cognitive load, and apparent sender familiarity all increase susceptibility. Arachchilage & Love (2014) proposed that effective security awareness programs must incorporate critical thinking, not just recognition skills — combining technology with education so users can independently detect and prevent threats.

---

### 2.3.2 Ethical Considerations in ML-Based Security Systems

The use of ML in security raises several ethical concerns:

**Bias and Fairness:** LLM-based detectors trained predominantly on English phishing emails may unfairly misclassify legitimate emails from non-English speakers or non-standard business contexts (Mehrabi et al., 2021). Mitigation strategies include diverse training sets, fair learning methods, and bias auditing across diverse test sets.

**Transparency and Explainability:** Users need to understand why emails are blocked. Explainability is also essential for building trust in automated systems (Ribeiro et al., 2016).

**Dual-Use Risk:** Research on phishing detection techniques could theoretically assist attackers in crafting more evasive emails. Responsible disclosure principles dictate that technical details be withheld and focus remain on defensive goals (Brundage et al., 2018).

**Environmental Impact:** Training large models like BERT and RoBERTa consumes significant energy (Strubell et al., 2019). Research should consider energy-efficient alternatives like DistilBERT, model compression, and transparent documentation of energy usage.

**Privacy:** Even with publicly available datasets, anonymization of email content and removal of personally identifiable information (PII) is essential to respect user privacy.

---

### 2.3.3 Research Gaps and Future Directions

Despite the progress of LLMs in phishing detection, several gaps remain:

- **Multilingual Detection:** Most research focuses on English-language phishing, despite phishing being a global threat. Multilingual and cross-cultural detectors are increasingly necessary (Lo et al., 2021).
- **Real-Time Deployment:** Most research addresses offline applications. Questions around real-world efficiency, model updating for emerging attacks, and integration with existing email security infrastructure remain underexplored (Sahingoz et al., 2019).
- **Adversarial Robustness:** LLM-based detectors can be evaded by attackers who craft emails specifically to bypass them. Adversarial training, ensemble methods, and adversarial detection mechanisms warrant further research.
- **User Education Integration:** The potential for LLMs to not only detect phishing but also educate users about threats in real time remains largely unexplored — a promising direction for combining automated protection with human awareness.

---

## References

- Abu-Nimeh, S., et al. (2007). A comparison of machine learning techniques for phishing detection. *eCrime Researchers Summit*, 60–69.
- Alkhalil, Z., et al. (2021). Phishing attacks: A recent comprehensive study. *Frontiers in Computer Science, 3*, 563060.
- Apruzzese, G., et al. (2018). On the effectiveness of machine and deep learning for cyber security. *CyCon*, 371–390.
- Arachchilage, N. A. G., & Love, S. (2014). Security awareness of computer users. *Computers in Human Behavior, 38*, 304–312.
- Ariyadasa, S. T., et al. (2023). Evaluation of pooling operations in convolutional architectures. *Engineering Applications of AI, 123*, 106349.
- Basit, A., et al. (2021). A comprehensive survey of AI-enabled phishing detection. *Telecommunication Systems, 76*(1), 139–154.
- Brown, T. B., et al. (2020). Language models are few-shot learners. *NeurIPS, 33*, 1877–1901.
- Brundage, M., et al. (2018). The malicious use of artificial intelligence. *arXiv:1802.07228*.
- Buczak, A. L., & Guven, E. (2016). A survey of data mining and ML methods for cyber security. *IEEE Communications Surveys & Tutorials, 18*(2), 1153–1176.
- Chandrasekaran, M., et al. (2006). Phishing email detection based on structural properties. *NYS Cyber Security Symposium*.
- Chiew, K. L., et al. (2018). A survey of phishing attacks. *Expert Systems with Applications, 106*, 1–20.
- Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL-HLT*, 4171–4186.
- Fette, I., et al. (2007). Learning to detect phishing emails. *WWW*, 649–656.
- Greshake, K., et al. (2023). Compromising LLM-integrated applications with indirect prompt injection. *ACM AISec*, 79–90.
- Gupta, B. B., et al. (2018). Fighting against phishing attacks. *Neural Computing and Applications, 28*(12), 3629–3654.
- Heiding, F., et al. (2024). Devising and detecting phishing emails using large language models. *IEEE Access*.
- Hong, J. (2012). The state of phishing attacks. *Communications of the ACM, 55*(1), 74–81.
- Jurafsky, D., & Martin, J. H. (2009). *Speech and Language Processing* (2nd ed.). Pearson.
- Khonji, M., et al. (2013). Phishing detection: A literature survey. *IEEE Communications Surveys & Tutorials, 15*(4), 2091–2121.
- Koide, T., et al. (2024). ChatSpamDetector: Leveraging LLMs for phishing email detection. *arXiv:2402.18093*.
- Kumaraguru, P., et al. (2010). Teaching Johnny not to fall for phish. *ACM TOIT, 10*(2), 1–31.
- Lastdrager, E. E. H. (2014). Achieving a consensual definition of phishing. *Crime Science, 3*(1), 9.
- Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv:1907.11692*.
- Liu, Y., et al. (2023). Prompt injection attack against LLM-integrated applications. *arXiv:2306.05499*.
- Liu, Y., et al. (2024). Formalizing and benchmarking prompt injection attacks. *USENIX Security 24*, 1831–1847.
- Lo, W. W., et al. (2021). E-GraphSAGE: A graph neural network-based intrusion detection system. *arXiv:2103.16329*.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS, 30*, 4765–4774.
- Mehrabi, N., et al. (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys, 54*(6), 1–35.
- Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. *arXiv:1301.3781*.
- Parsons, K., et al. (2014). Determining employee awareness using the HAIS-Q. *Computers & Security, 42*, 165–176.
- Pennington, J., et al. (2014). GloVe: Global vectors for word representation. *EMNLP*, 1532–1543.
- Ribeiro, M. T., et al. (2016). 'Why should I trust you?' Explaining any classifier. *KDD*, 1135–1144.
- Sahingoz, O. K., et al. (2019). Machine learning based phishing detection from URLs. *Expert Systems with Applications, 117*, 345–357.
- Salloum, S., et al. (2022). A systematic literature review on phishing email detection using NLP. *IEEE Access, 10*, 65703–65727.
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108*.
- Strubell, E., et al. (2019). Energy and policy considerations for deep learning in NLP. *ACL*, 3645–3650.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*, 5998–6008.
- Vishwanath, A., et al. (2011). Why do people get phished? *Decision Support Systems, 51*(3), 576–586.
- Xin, Y., et al. (2018). Machine learning and deep learning methods for cybersecurity. *IEEE Access, 6*, 35365–35381.
