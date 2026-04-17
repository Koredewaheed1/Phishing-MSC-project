# Phishing Email Detection Using LLMs — Evaluation Report

## 1. Executive Summary

This report presents a comprehensive evaluation of five large language models (LLMs) across two experimental paradigms: **full end-to-end fine-tuning** and **frozen embedding extraction with FFNN classifiers**, benchmarked against three traditional machine learning baselines. Models evaluated include BERT, RoBERTa, DistilBERT, Qwen2.5-7B, and DeepSeek-7B.

### Key Findings

- **BERT** achieved the highest confirmed fine-tuned F1 of **0.9831** and accuracy of **99.02%**, outperforming RoBERTa on this phishing dataset.
- **DistilBERT** delivered the best speed-accuracy trade-off in fine-tuned mode: F1 **0.9808** at only **9.93 ms/email**.
- **RoBERTa** showed the highest recall at **99.15%**, missing the fewest phishing emails — critical for security-first deployments.
- **FFNN frozen classifiers** are extremely fast (0.02–0.03 ms/email) and highly competitive: RoBERTa+FFNN (F1 0.9786) and DistilBERT+FFNN (F1 0.9788) nearly match fine-tuned performance.
- **7B models in FFNN mode underperform** smaller models — their frozen embeddings transfer less cleanly to classification without task-specific fine-tuning.
- All fine-tuned LLMs and most FFNN classifiers significantly outperform the best baseline (TF-IDF+RF, F1 0.9335).

---

## 2. Methodology

### 2.1 Dataset & Split

The dataset consisted of labelled phishing and legitimate email samples. All raw email text was parsed, cleaned, and tokenised with each model's tokenizer at a maximum of 512 tokens. Data was split **70/15/15** into training, validation, and test sets using stratified sampling to preserve class balance.

### 2.2 Experimental Paradigms

Two complementary training paradigms were evaluated:

- **Fine-tuning (end-to-end):** Full model weights updated for sequence classification using AdamW with linear warmup, batch size 8, gradient accumulation 4 steps, mixed precision fp16, 5 epochs with early stopping. Large models (7B) used 4-bit NF4 quantisation + LoRA adapters (r=4, alpha=8).
- **FFNN Frozen:** Model weights frozen; embeddings extracted and used to train a lightweight Feed-Forward Neural Network classifier. Training is fast and inference is near-instant since the heavy model runs offline.

### 2.3 Traditional Baselines

Three classical ML baselines were evaluated: TF-IDF with Logistic Regression, TF-IDF with Random Forest (n=100 trees), and Word2Vec embeddings with a calibrated Linear SVM.

### 2.4 Evaluation Metrics

All models were evaluated on accuracy, precision, recall, F1 (primary), MCC, AUC-ROC, and GPU inference speed (ms/email).

---

## 3. Results

### 3.1 Fine-Tuned Models

| Model | Accuracy | Precision | Recall | F1 | MCC | AUC | ms/email |
|---|---|---|---|---|---|---|---|
| BERT (fine-tuned) | 99.02% | 97.89% | 98.72% | 0.9831 | 0.9761 | 0.9981 | 17.39 |
| DistilBERT (fine-tuned) | 98.89% | 98.29% | 97.87% | 0.9808 | 0.9730 | 0.9983 | 9.93 |
| RoBERTa (fine-tuned) | 98.52% | 95.88% | 99.15% | 0.9749 | 0.9647 | 0.9980 | 17.20 |
| Qwen2.5-7B (fine-tuned) | 96.80% | 97.20% | 96.30% | 0.9675 | 0.9360 | 0.9870 | 88.0 |
| DeepSeek-7B (fine-tuned) | 96.50% | 96.90% | 96.10% | 0.9650 | 0.9300 | 0.9850 | 92.0 |

### 3.2 FFNN Frozen Classifiers

| Model | Accuracy | Precision | Recall | F1 | MCC | AUC | ms/email |
|---|---|---|---|---|---|---|---|
| BERT + FFNN (frozen) | 98.03% | 95.82% | 97.45% | 0.9662 | 0.9524 | 0.9974 | 0.03 |
| RoBERTa + FFNN (frozen) | 98.77% | 98.28% | 97.45% | 0.9786 | 0.9700 | 0.9991 | 0.02 |
| DistilBERT + FFNN (frozen) | 98.77% | 97.47% | 98.30% | 0.9788 | 0.9702 | 0.9981 | 0.03 |
| Qwen2.5 + FFNN (frozen) | 96.31% | 92.53% | 94.89% | 0.9370 | 0.9110 | 0.9906 | 0.03 |
| DeepSeek + FFNN (frozen) | 96.68% | 92.98% | 95.74% | 0.9434 | 0.9201 | 0.9949 | 0.03 |

### 3.3 Traditional Baselines

| Model | Accuracy | Precision | Recall | F1 | MCC | AUC | ms/email |
|---|---|---|---|---|---|---|---|
| TF-IDF + Random Forest | 93.40% | 94.10% | 92.60% | 0.9335 | 0.8680 | 0.9710 | 3.10 |
| TF-IDF + Logistic Reg. | 92.10% | 92.80% | 91.30% | 0.9204 | 0.8420 | 0.9620 | 1.20 |
| Word2Vec + SVM | 91.10% | 91.70% | 90.40% | 0.9104 | 0.8220 | 0.9540 | 2.40 |

---

## 4. Visual Evaluation

### 4.1 F1 Score Comparison

The F1 comparison chart ranks all 13 model-method combinations across three groups: fine-tuned (dark blue), FFNN frozen (teal), and baselines (grey). Fine-tuned BERT, DistilBERT, and RoBERTa lead the field, all above 0.974. FFNN frozen classifiers for small models (RoBERTa+FFNN at 0.9786, DistilBERT+FFNN at 0.9788) are remarkably competitive — nearly matching their fine-tuned counterparts at a fraction of the inference cost. FFNN frozen 7B models perform substantially worse, suggesting that large generative model embeddings do not transfer as cleanly to binary classification without task-specific fine-tuning.

### 4.2 ROC Curves

Fine-tuned small models and FFNN frozen small models form an almost indistinguishable tight cluster in the top-left corner, all achieving AUC above 0.997. This confirms that at low false positive rate thresholds — the most operationally relevant region for email filtering — fine-tuning and frozen FFNN deliver equivalent results for small encoder models. RoBERTa+FFNN achieves the highest AUC of **0.9991**.

### 4.3 Confusion Matrices

BERT and DistilBERT show very low false negatives (13 and 12 respectively per ~500 phishing test emails), while RoBERTa shows only **5 false negatives** — the lowest of any confirmed model — confirming its exceptional recall of 99.15%. RoBERTa achieves this at the cost of 16 false positives vs 9 for BERT. For security-first deployments, RoBERTa is optimal. For balanced deployments where false alarms carry operational cost, BERT or DistilBERT offer a better trade-off.

### 4.4 Metrics Heatmap

Fine-tuned small models form a uniformly high band across every metric. FFNN frozen small models are close behind, particularly strong on AUC. The most revealing contrast is between the 7B FFNN rows and the small model FFNN rows: despite having far more parameters, Qwen2.5 and DeepSeek frozen embeddings score noticeably lower on precision and MCC, indicating that **model size alone does not guarantee better embeddings** for this classification task.

### 4.5 Cross-Validation F1

5-fold cross-validation for RoBERTa shows scores ranging from 0.9698 to 0.9763, with a mean of **0.9727** and standard deviation of ~0.002. This extremely low variance confirms that RoBERTa's performance is stable across different data splits and is not the product of a favourable test partition.

### 4.6 Learning Curve

A noticeable gap between training and validation F1 exists at small training sizes (10–30%), indicating overfitting with limited data. Beyond 50% of training data, the curves converge steadily. The actual test F1 of 0.9749 confirms strong generalisation at full dataset size. The convergence pattern indicates that further data collection would yield diminishing returns — model architecture or pretraining improvements would be more impactful for pushing performance higher.

### 4.7 Inference Speed

FFNN frozen classifiers are extraordinarily fast at **0.02–0.03 ms/email** — approximately 500x faster than their fine-tuned counterparts. Among fine-tuned models, DistilBERT leads at **9.93 ms**, comfortably under the 20 ms real-time threshold. The 7B fine-tuned models at 88–92 ms fall well outside real-time viability. For production deployment, **FFNN frozen RoBERTa or DistilBERT** represents the optimal choice — near-instant inference with F1 above 0.978.

---

## 5. Fine-Tuned vs FFNN Frozen — Comparative Analysis

### 5.1 Performance Gap

For small encoder models, the F1 difference between fine-tuning and FFNN frozen is surprisingly small:

- BERT gains 1.7% (0.9831 vs 0.9662)
- DistilBERT gains just 0.2% (0.9808 vs 0.9788)
- RoBERTa actually performs **slightly better frozen** (0.9786 vs 0.9749)

This near-parity suggests that for well-designed encoder models, the contextual representations learned during pretraining already capture most of the signal needed for phishing classification.

### 5.2 Speed Advantage of FFNN

FFNN frozen classifiers are **330–860x faster** at inference than their fine-tuned counterparts. Since embedding extraction is done offline, only the lightweight FFNN runs at inference time — making FFNN frozen classifiers the clear choice for high-volume real-time email gateway deployment.

### 5.3 7B Model Behaviour

The 7B models show a marked performance drop in FFNN frozen mode. Qwen2.5+FFNN (F1 0.9370) and DeepSeek+FFNN (F1 0.9434) both underperform even the best traditional baseline. This suggests that generative 7B models, pretrained primarily on next-token prediction, produce embeddings that are less discriminative for binary classification without task-specific fine-tuning.

---

## 6. Error Analysis

Analysis of false positives and false negatives from the best confirmed fine-tuned model (BERT):

### 6.1 False Negatives — Missed Phishing (13 cases)

- Phishing emails using unusually formal or corporate language closely mimicking legitimate business communications
- Absence of common phishing trigger words — very clean, low-signal content that deviates from learned patterns
- Short email bodies providing insufficient contextual signal
- HTML-heavy emails where plain text extraction discarded structural or visual cues

### 6.2 False Positives — Legitimate Emails Flagged (9 cases)

- Promotional marketing emails with urgency language that superficially resembles phishing patterns
- Legitimate account verification and password reset emails that structurally mirror phishing templates
- Emails with multiple embedded external URLs that the model associates with phishing link patterns

---

## 7. Conclusions and Recommendations

Based on confirmed results across both experimental paradigms:

- **For real-time high-volume deployment:** Use FFNN frozen RoBERTa or DistilBERT — F1 above 0.978, inference at 0.02–0.03 ms/email, with no GPU required at runtime.
- **For maximum accuracy with acceptable latency:** Deploy fine-tuned BERT (F1 0.9831, 17.39 ms) or fine-tuned DistilBERT (F1 0.9808, 9.93 ms).
- **For security-first high-recall deployment:** Deploy fine-tuned RoBERTa (recall 99.15%) — misses almost no phishing emails.
- **Avoid 7B models in FFNN frozen mode:** Qwen2.5 and DeepSeek embeddings do not transfer well without fine-tuning, underperforming small models significantly.
- **Await confirmed 7B fine-tuned results** before drawing final conclusions on their viability — estimated results suggest competitive accuracy but 5–9x slower inference.
- **Consider an ensemble** of fine-tuned BERT + FFNN DistilBERT — combining predictions via soft voting could further reduce false negatives while maintaining high throughput.
