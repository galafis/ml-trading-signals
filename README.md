# 🤖 Ml Trading Signals

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-0194E2.svg)](https://mlflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit-learn-1.4-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](#english) | [Português](#português)

---

## English

### 🎯 Overview

**Ml Trading Signals** — Machine learning system for generating trading signals using XGBoost, LightGBM, and ensemble methods. Features 40+ technical indicators, MLflow tracking, and FastAPI inference API.

Total source lines: **2,204** across **22** files in **1** language.

### ✨ Key Features

- **Production-Ready Architecture**: Modular, well-documented, and following best practices
- **Comprehensive Implementation**: Complete solution with all core functionality
- **Clean Code**: Type-safe, well-tested, and maintainable codebase
- **Easy Deployment**: Docker support for quick setup and deployment

### 🚀 Quick Start

#### Prerequisites
- Python 3.12+
- Docker and Docker Compose (optional)

#### Installation

1. **Clone the repository**
```bash
git clone https://github.com/galafis/ml-trading-signals.git
cd ml-trading-signals
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

#### Running

```bash
python src/api/main.py
```

## 🐳 Docker

```bash
# Build the image
docker build -t ml-trading-signals .

# Run the container
docker run -p 8000:8000 ml-trading-signals
```

### 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run with verbose output
pytest -v
```

### 📁 Project Structure

```
ml-trading-signals/
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
├── docs/
│   └── images/
├── examples/
│   └── predict_signals.py
├── models/
│   └── README.md
├── notebooks/
│   ├── 01_quick_start.md
│   └── README.md
├── scripts/
│   └── generate_charts.py
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── data_preparation.py
│   │   └── technical_indicators.py
│   ├── inference/
│   │   └── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_pipeline.py
│   ├── utils/
│   │   └── __init__.py
│   └── __init__.py
├── tests/
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_api.py
│   │   └── test_training.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_features.py
│   │   └── test_models.py
│   └── __init__.py
├── CONTRIBUTING.md
├── README.md
├── pytest.ini
├── requirements.txt
└── train.py
```

### 🛠️ Tech Stack

| Technology | Usage |
|------------|-------|
| Python | 22 files |

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 👤 Author

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)

---

## Português

### 🎯 Visão Geral

**Ml Trading Signals** — Machine learning system for generating trading signals using XGBoost, LightGBM, and ensemble methods. Features 40+ technical indicators, MLflow tracking, and FastAPI inference API.

Total de linhas de código: **2,204** em **22** arquivos em **1** linguagem.

### ✨ Funcionalidades Principais

- **Arquitetura Pronta para Produção**: Modular, bem documentada e seguindo boas práticas
- **Implementação Completa**: Solução completa com todas as funcionalidades principais
- **Código Limpo**: Type-safe, bem testado e manutenível
- **Fácil Implantação**: Suporte Docker para configuração e implantação rápidas

### 🚀 Início Rápido

#### Pré-requisitos
- Python 3.12+
- Docker e Docker Compose (opcional)

#### Instalação

1. **Clone the repository**
```bash
git clone https://github.com/galafis/ml-trading-signals.git
cd ml-trading-signals
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

#### Execução

```bash
python src/api/main.py
```

### 🧪 Testes

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run with verbose output
pytest -v
```

### 📁 Estrutura do Projeto

```
ml-trading-signals/
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
├── docs/
│   └── images/
├── examples/
│   └── predict_signals.py
├── models/
│   └── README.md
├── notebooks/
│   ├── 01_quick_start.md
│   └── README.md
├── scripts/
│   └── generate_charts.py
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── data_preparation.py
│   │   └── technical_indicators.py
│   ├── inference/
│   │   └── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_pipeline.py
│   ├── utils/
│   │   └── __init__.py
│   └── __init__.py
├── tests/
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_api.py
│   │   └── test_training.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_features.py
│   │   └── test_models.py
│   └── __init__.py
├── CONTRIBUTING.md
├── README.md
├── pytest.ini
├── requirements.txt
└── train.py
```

### 🛠️ Stack Tecnológica

| Tecnologia | Uso |
|------------|-----|
| Python | 22 files |

### 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

### 👤 Autor

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)
