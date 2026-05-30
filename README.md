# Respiratory Disease Classification using CNN-LSTM Hybrid

Automated classification of respiratory diseases from lung sound recordings 
using a comparative study of CNN, LSTM, and hybrid CNN-LSTM deep learning architectures.

**Dataset:** ICBHI 2017 Respiratory Sound Database — 920 recordings, 126 patients, 5.5 hours  
**Disease Classes:** Normal · Pneumonia · COPD · Asthma · Bronchitis  
**Deployment:** Interactive Streamlit web application  
**Team:** Sneha Rathod · Shravani Chavan · Vaidehi Jadhav  
**Guide:** Dr. Venkat Patil | Smt. Indira Gandhi College of Engineering, Ghansoli

---

## Model Performance

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | AUC-ROC |
|---|---|---|---|---|---|
| CNN (Standalone) | ~89% | 0.87 | 0.86 | 0.86 | 0.91 |
| LSTM (Standalone) | ~91% | 0.89 | 0.90 | 0.89 | 0.93 |
| CNN-LSTM Hybrid | ~94% | 0.93 | 0.92 | 0.92 | 0.96 |

### Per-Class Results — CNN-LSTM Model

| Disease Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Normal | 0.97 | 0.95 | 0.96 |
| COPD | 0.93 | 0.92 | 0.92 |
| Pneumonia | 0.91 | 0.90 | 0.90 |
| Asthma | 0.92 | 0.91 | 0.91 |
| Bronchitis | 0.88 | 0.86 | 0.87 |

---

## System Pipeline
