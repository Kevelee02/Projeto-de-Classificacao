Projeto de **Classificação Supervisionada** para prever probabilidade de atraso em cards/entregas a partir de dados operacionais e logs de execução.

## 🎯 Objetivo

Construir um modelo preditivo capaz de identificar antecipadamente entregas com risco de atraso, auxiliando priorização e gestão operacional.

Problema tratado como:

- Binary Classification
- Imbalanced Classification
- Risk Prediction
- Early Warning Modeling

Target:

```python id="r4fa8n"
Atrasou
0 = Não atrasou
1 = Atrasou
```

---

# 🚀 Pipeline do Projeto

## 1. Construção da Base Analítica

Integração de múltiplas fontes:

- Cards
- Logs operacionais
- Checklists
- Itens de checklist

Processos realizados:

- Data cleaning
- Join/Merge entre tabelas
- Agregações por card
- Tratamento de missing values
- Criação da tabela mestre analítica

---

## 2. Feature Engineering

Criação de variáveis preditoras:

### Variáveis temporais
- Dias até vencimento do card
- Dias até limite do planejamento
- Dia da semana do vencimento
- Dia da semana do planejamento
- Flag de vencimento em fim de semana

## 📊 Modelos Avaliados

Foram comparados múltiplos algoritmos:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- XGBoost + GridSearchCV

---

## ⚙️ Técnicas Aplicadas

### Machine Learning
- Supervised Learning
- Binary Classification
- Ensemble Models
- Tree-Based Models

### Tratamento de Desbalanceamento
- Class Weights
- Dataset Balancing
- Comparative balanced vs imbalanced analysis

### Model Selection
- Hyperparameter Tuning
- Grid Search Cross Validation

---

## 📈 Métricas de Avaliação

Modelos avaliados com foco em detecção de atraso:

- Precision
- Recall
- ROC AUC
- Classification Report
- Curva ROC



OBS: Algumas tabelas foram Excluidas por questões de privacidade
