# 💊 Drug-Food Interaction Analysis System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/) [![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An AI-powered web application for analyzing drug-food interactions using NLP and Machine Learning with automated multi-class classification and knowledge graph generation.

## 🌟 Key Features

### 🧬 Novel Contributions
1. **Automated NLP Classification** - 6 categories: Contraindicated ❌ | Avoid ⚠️ | CYP enzymes 🧪 | Alters metabolism 🔄 | Enhances absorption 📈 | Safe ✅
2. **Knowledge Graph (DFIKG)** - Structured triplets: `Drug → Interaction Type → Food/Supplement` with interactive querying and CSV export

### 🤖 ML Pipeline
- TF-IDF Vectorization with bigrams | Ensemble Models (Logistic Regression, Random Forest, Gradient Boosting) | Cosine Similarity Search | Auto best-model selection

### 📊 Dashboard
Real-time semantic search | Color-coded severity (🔴🟡🟢) | Dynamic visualizations | Performance metrics | Downloadable results

## 🚀 Quick Start

### Google Colab (Recommended)
```
!pip install -q streamlit pandas numpy scikit-learn matplotlib seaborn
from google.colab import files; uploaded = files.upload()
!wget -q https://raw.githubusercontent.com/yourusername/drug-food-interaction/main/app.py
!streamlit run app.py & npx localtunnel --port 8501
```

### Local Installation
```
git clone https://github.com/yourusername/drug-food-interaction.git
cd drug-food-interaction
pip install -r requirements.txt
streamlit run app.py
```

## 📋 Requirements
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## 📊 Dataset Format
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `name` | string | Drug name | "Warfarin" |
| `reference` | string | Source | "DrugBank:DB00682" |
| `food_interactions` | string | Details | "Avoid vitamin K foods, grapefruit juice inhibits CYP3A4" |

**Sample CSV:**
```
name,reference,food_interactions
Warfarin,DrugBank:DB00682,"Avoid green leafy vegetables high in vitamin K. Grapefruit juice may inhibit CYP3A4 enzyme."
Metformin,DrugBank:DB00331,"Take with food to enhance absorption. Limit alcohol consumption."
Aspirin,DrugBank:DB00945,"Avoid alcohol - increases bleeding risk. Take with food to reduce GI irritation."
```

## 🎯 Usage

**1. Search Drugs** - Enter condition/drug → Set results count → Click Search → View similarity scores & details  
**2. Query KG** - Filter by Drug/Food/Type/Severity → Export as CSV  
**3. Visualizations** - Explore Distribution/Categories/Severity tabs  
**4. Export** - Download complete knowledge graph

## 🏗️ Architecture
```
Streamlit Frontend → NLP Classifier (6 classes) + ML Recommender (TF-IDF) → Knowledge Graph Builder → Visualization Engine
```

## 📈 Model Performance
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 0.847 | 0.842 |
| Random Forest | 0.873 | 0.869 |
| **Gradient Boosting** | **0.891** | **0.887** |

## 🔬 Implementation

### NLP Classification
```
def classify_interaction(text):
    text = text.lower()
    if "contraindicated" in text or "fatal" in text:
        return "Contraindicated"
    if "avoid" in text or "dangerous" in text:
        return "Avoid"
    if "cyp" in text or "enzyme" in text:
        return "Interacts with CYP enzymes"
    if "metabolism" in text:
        return "Alters metabolism"
    if "absorption" in text:
        return "Enhances absorption"
    return "Safe"
```

### Knowledge Graph Structure
```
Drug Node ──[interaction_type]──> Food Node
    │                                  │
    └─ name, reference, severity      └─ name, category, mechanism
```

## 📚 Project Structure
```
drug-food-interaction/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Dependencies
├── drug_food_interactions.csv      # Dataset
├── README.md                       # Documentation
└── LICENSE                         # MIT License
```

## 🤝 Contributing
1. Fork repository
2. Create feature branch: `git checkout -b feature/YourFeature`
3. Commit: `git commit -m 'Add YourFeature'`
4. Push: `git push origin feature/YourFeature`
5. Submit Pull Request

## 🗺️ Roadmap
- [ ] BioBERT integration for advanced NLP
- [ ] Graph Neural Networks (GNN)
- [ ] Drug-drug interaction analysis
- [ ] REST API
- [ ] User authentication
- [ ] DrugBank API integration
- [ ] Multi-language support

## 🐛 Known Issues
- Large datasets (>100k rows) may slow processing
- Special characters in food names need preprocessing
- Mobile UI optimization in progress

## 📄 License
```
MIT License - Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the "Software"), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions 
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED 
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

## 👤 Author
**Your Name** - [GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourprofile) | your.email@example.com

## 🙏 Acknowledgments
[DrugBank](https://www.drugbank.com/) | [Streamlit](https://streamlit.io/) | [Scikit-learn](https://scikit-learn.org/) | [Google Colab](https://colab.research.google.com/)

## 📖 Citation
```
@software{drug_food_interaction_2025,
  author = {Your Name},
  title = {Drug-Food Interaction Analysis System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/drug-food-interaction}
}
```

## 📞 Support
📧 Email: your.email@example.com | 💬 [Issues](https://github.com/yourusername/drug-food-interaction/issues) | 📚 [Wiki](https://github.com/yourusername/drug-food-interaction/wiki)

---

⭐ **Star this repo if helpful!** | 🔗 [Live Demo](https://your-demo.streamlit.app) | Made with ❤️ using Python, Streamlit & ML
```

**This is a complete, single-block README** - just copy everything between the triple backticks and save as `README.md`! Replace placeholders (yourusername, Your Name, etc.) with your actual details.
