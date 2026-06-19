# 🧠 Low-Resource Language Reasoning Framework

An evaluation framework designed to measure how effectively Large Language Models (LLMs) reason and translate across low-resource languages. This project explores whether structured prompting strategies and linguistic guidance improve model performance on languages with limited training data.

The framework evaluates model outputs using automated scoring techniques and compares reasoning quality through translation and question-answering tasks.

---

## 🧠 Technologies

### AI / NLP

* Python
* LLaMA 3.1
* Large Language Models (LLMs)
* Natural Language Processing (NLP)
* Prompt Engineering
* Chain-of-Thought Prompting

### Evaluation & Metrics

* SentenceTransformers
* BERTScore
* Cosine Similarity
* Semantic Similarity Evaluation

### Data & Visualization

* Pandas
* Matplotlib
* Regular Expressions (Regex)

### Research Concepts

* Low-Resource Languages
* Translation Evaluation
* Interlinear Glossed Text (IGT)
* Morphological Analysis

---

## ⚙️ Features

### 🌍 Low-Resource Language Evaluation

Evaluates model performance across languages with limited available training data.

### 🧠 Glossary-Guided Reasoning

Provides linguistic gloss information to guide translation and reasoning tasks.

### 📝 Multiple Evaluation Tasks

Measures performance through:

* Multiple-choice grammar questions
* Open-ended translation tasks

### 📊 Automated Performance Scoring

Generates evaluation metrics automatically using:

* Accuracy
* Cosine Similarity
* BERTScore

### 📈 Visualization & Reporting

Creates plots and CSV summaries to compare language performance.

### 🔄 Repeatable Benchmarking Pipeline

Runs the same evaluation process across multiple datasets and languages.

---

## 🔧 The Process

The project was built as a structured evaluation pipeline:

### 1. Dataset Preparation

Language datasets were prepared using:

* Multiple-choice grammar questions
* Interlinear glossed text (IGT)
* Ground-truth translations

### 2. Prompt Construction

Two prompting approaches were implemented:

* Direct multiple-choice prompting
* Glossary-guided Chain-of-Thought prompting

### 3. Model Inference

Prompts were sent to an LLM to generate predictions and translations.

### 4. Evaluation

Outputs were evaluated using:

* Accuracy for multiple-choice tasks
* Cosine similarity for semantic comparison
* BERTScore for translation quality

### 5. Reporting

Results were exported into:

* CSV summary files
* Visualization charts
* Aggregate evaluation reports

---

## 📚 What I Learned

* How to design evaluation frameworks for LLMs
* Applying prompt engineering techniques to structured reasoning tasks
* Measuring language understanding beyond simple accuracy
* Working with embedding-based similarity metrics
* Building automated benchmarking pipelines
* Understanding challenges associated with low-resource languages

---

## 🚀 How It Can Be Improved

* Compare additional LLM architectures
* Expand evaluation across more languages
* Add human evaluation for translation quality
* Introduce retrieval-augmented prompting
* Support multilingual embeddings
* Add experiment tracking and dashboards
* Build an interactive interface for running evaluations

---

## ▶️ Running the Project

### 1. Clone the repository

```bash
git clone <repo-url>
cd low-resource-language-reasoning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API credentials

Create an environment file:

```bash
API_KEY=your_api_key
```

### 4. Add datasets

Place language datasets in the project directory:

```text
language_mcq.txt
language_igt.txt
```

### 5. Run evaluation

```bash
python main.py
```

### 6. View outputs

Generated results will appear inside:

```text
outputs/
```

Including:

* CSV summaries
* Evaluation metrics
* Performance plots

---

## 📌 Notes

This project focuses on evaluating reasoning behavior rather than training models. Results may vary across models, prompts, and language families.
