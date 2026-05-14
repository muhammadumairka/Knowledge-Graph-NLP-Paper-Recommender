# Knowledge Graph + NLP Paper Recommender

## Overview

This project implements a hybrid AI system for scientific paper recommendation by combining:

* Natural Language Processing (NLP)
* Knowledge Graph Analysis
* Graph-Based Ranking Algorithms
* Semantic Similarity using Transformer Embeddings

The system analyzes scientific paper abstracts, generates semantic embeddings, constructs a graph representation of relationships between papers, and ranks relevant papers using a hybrid scoring mechanism.

This project was developed to explore practical applications of:

* Knowledge Graphs
* Scientific Literature Analysis
* NLP-based Semantic Search
* Graph Machine Learning concepts

---

# Key Features

* Semantic similarity analysis using transformer embeddings
* Knowledge graph construction using NetworkX
* Graph-based ranking with PageRank and centrality measures
* Hybrid recommendation scoring
* Research-oriented AI pipeline
* Lightweight and easy to extend

---

# Technologies Used

| Technology            | Purpose                         |
| --------------------- | ------------------------------- |
| Python                | Core development                |
| Sentence-Transformers | Semantic embeddings             |
| NetworkX              | Graph construction and analysis |
| Scikit-learn          | Similarity computation          |
| Pandas                | Data handling                   |

---

# System Architecture

```text
+--------------------+
| Research Papers    |
| (CSV Dataset)      |
+----------+---------+
           |
           v
+--------------------+
| Text Preprocessing |
+----------+---------+
           |
           v
+------------------------------+
| Sentence Transformer Model   |
| (Semantic Embeddings)        |
+----------+-------------------+
           |
           v
+------------------------------+
| Similarity Computation       |
| (Cosine Similarity)          |
+----------+-------------------+
           |
           v
+------------------------------+
| Knowledge Graph Construction |
| (NetworkX)                   |
+----------+-------------------+
           |
           v
+------------------------------+
| Graph Ranking Algorithms     |
| (PageRank / Centrality)      |
+----------+-------------------+
           |
           v
+------------------------------+
| Recommendation Engine        |
+----------+-------------------+
           |
           v
+------------------------------+
| Top Relevant Papers          |
+------------------------------+
```

---

# Repository Structure

```text
Knowledge-Graph-NLP-Paper-Recommender/
│
├── main.py
├── data.csv
├── requirements.txt
└── README.md
```

---

# Installation Instructions

## 1. Clone Repository

```bash
git clone https://github.com/muhammadumairka/Knowledge-Graph-NLP-Paper-Recommender.git
cd Knowledge-Graph-NLP-Paper-Recommender
```

---

## 2. Create Virtual Environment (Optional but Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the Project

```bash
python main.py
```

---

# Sample Input

```text
Enter research topic:
Machine learning in healthcare
```

---

# Sample Output

```text
Top Recommendations:

1. Machine Learning for Healthcare
2. AI in Education
3. Natural Language Processing Advances
```

---

# Methodology

The recommendation workflow follows these stages:

1. Research paper abstracts are loaded from a dataset.
2. Transformer-based sentence embeddings are generated.
3. Cosine similarity is computed between papers.
4. A knowledge graph is constructed using similarity relationships.
5. PageRank and graph centrality metrics are applied.
6. Semantic similarity and graph ranking scores are combined.
7. Top-ranked papers are returned as recommendations.

---

# Research Motivation

Scientific literature is growing rapidly, making it difficult for researchers to identify influential and relevant papers efficiently.

This project explores how:

* NLP
* Semantic embeddings
* Graph analysis
* Knowledge extraction

can be integrated into intelligent research-support systems.

---

# Future Work

Planned improvements include:

* Integration with Large Language Models (LLMs)
* Real-world datasets from Semantic Scholar or arXiv
* Graph Neural Networks (GNNs)
* Interactive web interface
* Multi-language scientific recommendation support
* Citation prediction and trend analysis
* Retrieval-Augmented Generation (RAG) integration

---

# Author

Muhammad Umair

Machine Learning Researcher | Knowledge Graphs | NLP | Scientific Literature Analysis | Generative AI

GitHub: [https://github.com/muhammadumairka](https://github.com/muhammadumairka)
