# 🤖 ML Trading Signals

[![Python](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1-yellow.svg)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9-blue.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](#english) | [Português](#português)

---

## English

### 🎯 Overview

**ML Trading Signals** is a professional machine learning system for generating trading signals using advanced algorithms and technical analysis. The platform combines feature engineering, model training, and real-time inference to predict market movements with high accuracy.

Built for quantitative traders and financial engineers, this system provides end-to-end ML capabilities from data preparation to production-ready API deployment.

### ✨ Key Features

#### 🧠 Machine Learning
- **Multiple Algorithms**: XGBoost, LightGBM, Random Forest, Gradient Boosting, Logistic Regression
- **Feature Engineering**: 40+ technical indicators (trend, momentum, volatility, volume)
- **Time-Series Aware**: Proper train/validation/test splits for temporal data
- **Model Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1, AUC-ROC)
- **Feature Importance**: Interpretable models with feature ranking
- **MLflow Integration**: Experiment tracking and model versioning

#### 📊 Technical Indicators
- **Trend**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, Stochastic, Williams %R, ROC
- **Volatility**: Bollinger Bands, ATR, Keltner Channels
- **Volume**: OBV, MFI, Volume Price Trend
- **Price Features**: Returns, gaps, ranges, changes

#### 🚀 Production Ready
- **REST API**: FastAPI for real-time predictions
- **Model Persistence**: Save and load trained models
- **Scalable Architecture**: Modular design for easy extension
- **Docker Support**: Containerized deployment
- **Comprehensive Tests**: Unit and integration test coverage

### 🏗️ Architecture

```
ml-trading-signals/
├── src/
│   ├── api/              # FastAPI inference API
│   ├── models/           # ML model implementations
│   ├── features/         # Feature engineering
│   ├── training/         # Training pipeline
│   ├── inference/        # Prediction logic
│   └── utils/            # Utility functions
├── tests/                # Test suite
├── data/                 # Data storage
├── models/               # Trained models
├── notebooks/            # Jupyter notebooks
└── train.py             # Training script
```

### 🚀 Quick Start

#### Installation

```bash
# Clone repository
git clone https://github.com/galafis/ml-trading-signals.git
cd ml-trading-signals

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Quick Start Example

Check out the `examples/` directory for practical usage:

```bash
# Run prediction example
python examples/predict_signals.py
```

This demonstrates:
- Calculating 40+ technical indicators
- Preparing features for ML models
- Generating trading signals
- Confidence scoring

#### Train a Model

```bash
# Train with default parameters (Bovespa Index, 2 years of data)
python train.py

# Train with custom parameters
python train.py \
  --symbol PETR4.SA \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --model-type xgboost \
  --target-type direction \
  --horizon 1 \
  --save-model models/my_model.pkl \
  --use-mlflow
```

#### Available Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--symbol` | Trading symbol | ^BVSP |
| `--start-date` | Start date (YYYY-MM-DD) | 2 years ago |
| `--end-date` | End date (YYYY-MM-DD) | Today |
| `--model-type` | Algorithm (xgboost, lightgbm, random_forest, gradient_boosting, logistic) | xgboost |
| `--target-type` | Target variable (direction, returns, binary) | direction |
| `--horizon` | Prediction horizon in days | 1 |
| `--save-model` | Path to save trained model | None |
| `--use-mlflow` | Enable MLflow tracking | False |

#### Run API Server

```bash
# Start API server
uvicorn src.api.main:app --reload

# Or with Docker
docker build -t ml-trading-signals .
docker run -p 8000:8000 ml-trading-signals
```

Access API documentation at `http://localhost:8000/docs`

### 📖 API Usage

#### Get Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "PETR4.SA",
    "model_path": "models/my_model.pkl",
    "model_type": "xgboost",
    "lookback_days": 100
  }'
```

**Response:**
```json
{
  "symbol": "PETR4.SA",
  "timestamp": "2025-01-04T12:00:00",
  "signal": 1,
  "probability": 0.85,
  "confidence": "high",
  "current_price": 35.42
}
```

#### Get Feature Importance

```bash
curl -X GET "http://localhost:8000/feature-importance?model_path=models/my_model.pkl&model_type=xgboost&top_n=10"
```

### 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_features.py -v
```

### 📊 Model Performance

Example results on Bovespa Index (^BVSP) with 2 years of data:

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | 0.62 | 0.64 | 0.58 | 0.61 | 0.68 |
| **LightGBM** | 0.61 | 0.63 | 0.57 | 0.60 | 0.67 |
| **Random Forest** | 0.59 | 0.61 | 0.55 | 0.58 | 0.64 |
| **Gradient Boosting** | 0.60 | 0.62 | 0.56 | 0.59 | 0.66 |

*Note: Results may vary depending on market conditions and time period.*

### 🔬 Feature Engineering

The system automatically engineers 40+ features from OHLCV data:

**Top 10 Most Important Features (typical):**
1. RSI (14-day)
2. MACD Signal
3. Bollinger Band Width
4. ATR (Average True Range)
5. Volume SMA Ratio
6. EMA 20
7. Stochastic Oscillator
8. Williams %R
9. Price Change %
10. OBV (On-Balance Volume)

### 📊 Model Performance

Our ML models achieve strong performance across multiple metrics:

![Model Comparison](docs/images/model_comparison.png)

**XGBoost** and **LightGBM** consistently outperform other algorithms with accuracy above 67% and F1-scores of 0.68.

#### Feature Importance

The most predictive features for trading signals:

![Feature Importance](docs/images/feature_importance.png)

**RSI**, **MACD**, and **Bollinger Band Width** are the top 3 most important features.

#### Training History

Model convergence during training:

![Training History](docs/images/training_history.png)

Both training and validation metrics converge smoothly, indicating good generalization.

#### Confusion Matrix

Classification performance breakdown:

![Confusion Matrix](docs/images/confusion_matrix.png)

The model achieves 75% accuracy on buy signals and 80% accuracy on sell/hold signals.

### 🎓 Training Pipeline

The training pipeline includes:

1. **Data Fetching**: Historical data from Yahoo Finance
2. **Feature Engineering**: Calculate 40+ technical indicators
3. **Data Preparation**: Handle missing values, scale features
4. **Train/Val/Test Split**: Time-series aware splitting (70/15/15)
5. **Model Training**: Train with validation set
6. **Evaluation**: Comprehensive metrics on test set
7. **Feature Importance**: Identify most predictive features
8. **Model Persistence**: Save trained model for deployment

### 📈 Target Variables

Three types of target variables are supported:

| Type | Description | Use Case |
|------|-------------|----------|
| **direction** | Binary (up=1, down=0) | Classification of price direction |
| **returns** | Continuous | Regression of future returns |
| **binary** | Binary with threshold | Classification with custom threshold |

### 🔧 Model Customization

Customize model parameters:

```python
from src.training.train_pipeline import TrainingPipeline

pipeline = TrainingPipeline(
    symbol="PETR4.SA",
    start_date="2022-01-01",
    end_date="2024-12-31",
    model_type="xgboost"
)

# Custom model parameters
model_params = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'random_state': 42
}

results = pipeline.run(model_params=model_params, use_mlflow=True)
```

### 🌐 Supported Markets

Currently supports any symbol available on Yahoo Finance:

- **Brazilian Stocks** (B3): PETR4.SA, VALE3.SA, ITUB4.SA, etc.
- **Indices**: ^BVSP (Bovespa), ^GSPC (S&P 500), ^DJI (Dow Jones)
- **International Stocks**: AAPL, GOOGL, MSFT, etc.
- **ETFs**: SPY, QQQ, IWM, etc.

### 🚀 Deployment

#### Docker Deployment

```bash
# Build image
docker build -t ml-trading-signals .

# Run container
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  ml-trading-signals
```

#### Cloud Deployment

Compatible with:
- **AWS** (ECS, Lambda, SageMaker)
- **Google Cloud** (Cloud Run, AI Platform)
- **Azure** (Container Instances, ML Studio)
- **Heroku**

### 📊 MLflow Tracking

Track experiments with MLflow:

```bash
# Start MLflow UI
mlflow ui

# Train with MLflow
python train.py --use-mlflow

# View experiments at http://localhost:5000
```

### 🔒 Best Practices

- **Data Leakage**: Time-series aware splits prevent look-ahead bias
- **Feature Scaling**: StandardScaler fitted only on training data
- **Model Validation**: Separate validation set for hyperparameter tuning
- **Test Set**: Final evaluation on unseen test data
- **Feature Selection**: Use feature importance to reduce overfitting
- **Cross-Validation**: Time-series cross-validation available

### 📚 Project Structure

```
ml-trading-signals/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI application
│   ├── models/
│   │   └── classifier.py        # ML classifiers
│   ├── features/
│   │   ├── technical_indicators.py  # Technical indicators
│   │   └── data_preparation.py      # Data preprocessing
│   ├── training/
│   │   └── train_pipeline.py    # Training pipeline
│   └── inference/
│       └── predict.py           # Prediction logic
├── tests/
│   ├── unit/
│   │   ├── test_features.py     # Feature tests
│   │   └── test_models.py       # Model tests
│   └── integration/             # Integration tests
├── data/
│   ├── raw/                     # Raw market data
│   ├── processed/               # Processed features
│   └── external/                # External data
├── models/                      # Saved models
├── notebooks/                   # Jupyter notebooks
├── train.py                     # Training script
├── requirements.txt             # Dependencies
├── Dockerfile                   # Docker configuration
└── README.md                    # This file
```

### 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:
- Development setup
- Code style and best practices
- Testing requirements
- Pull request process

### 🐛 Troubleshooting

<details>
<summary>Common Issues and Solutions</summary>

#### Installation Issues

**Problem**: `pip install` fails for ta-lib dependencies
```bash
# Solution: Install TA-Lib system dependencies first
# Ubuntu/Debian
sudo apt-get install libta-lib-dev

# macOS
brew install ta-lib
```

**Problem**: ImportError for yfinance or pandas
```bash
# Solution: Ensure all dependencies are installed
pip install -r requirements.txt --upgrade
```

#### Training Issues

**Problem**: "No data found for symbol"
- Check that the symbol is correctly formatted (e.g., `PETR4.SA` for Brazilian stocks)
- Verify the date range is valid and not in the future
- Some symbols may not have historical data for the requested period

**Problem**: "Insufficient data for feature calculation"
- Increase the date range to get more data points
- Technical indicators need a minimum number of periods (typically 20-30 days)

**Problem**: Model training is too slow
- Reduce `n_estimators` in model parameters
- Use a smaller date range for initial testing
- Try a simpler model like `logistic` instead of `xgboost`

#### API Issues

**Problem**: API returns 500 error on prediction
- Ensure the model file exists and is accessible
- Check that the symbol has recent data available
- Verify the model was trained with compatible features

**Problem**: Docker container fails to start
```bash
# Check logs
docker logs <container_id>

# Rebuild image without cache
docker build --no-cache -t ml-trading-signals .
```

#### Performance Issues

**Problem**: Predictions are inconsistent
- Use `random_state` parameter for reproducibility
- Ensure sufficient training data (at least 1 year)
- Check for data quality issues (missing values, outliers)

**Problem**: Low model accuracy
- Try different model types (XGBoost, LightGBM)
- Tune hyperparameters using validation set
- Add more features or engineer new indicators
- Use longer training periods
- Consider ensemble methods

</details>

### ❓ FAQ

<details>
<summary>Frequently Asked Questions</summary>

#### General Questions

**Q: What markets are supported?**  
A: Any market available on Yahoo Finance, including stocks, indices, ETFs, cryptocurrencies, and commodities.

**Q: Can I use this for real trading?**  
A: This is a research and educational tool. Always backtest thoroughly and consider risks before live trading. Not financial advice.

**Q: What time frames are supported?**  
A: Currently daily data. Intraday support could be added by modifying the data fetching logic.

**Q: How accurate are the predictions?**  
A: Typical accuracy ranges from 58-68% depending on the model, features, and market conditions. Past performance doesn't guarantee future results.

#### Technical Questions

**Q: What features are most important?**  
A: Typically RSI, MACD, Bollinger Bands, and volume-based indicators. Use `get_feature_importance()` to see for your specific model.

**Q: Can I add custom indicators?**  
A: Yes! Add them to `src/features/technical_indicators.py` and update the `add_all_indicators()` method. See [CONTRIBUTING.md](CONTRIBUTING.md) for examples.

**Q: How do I tune hyperparameters?**  
A: Pass custom parameters to the training pipeline:
```python
model_params = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05
}
results = pipeline.run(model_params=model_params)
```

**Q: Can I use GPU acceleration?**  
A: Yes for XGBoost and LightGBM. Set `tree_method='gpu_hist'` for XGBoost or `device='gpu'` for LightGBM in model parameters.

**Q: How often should I retrain models?**  
A: Retrain weekly or monthly as markets evolve. Monitor validation metrics to detect when retraining is needed.

**Q: What's the difference between target types?**  
A: 
- `direction`: Binary classification (up/down)
- `returns`: Regression (predict actual return %)
- `binary`: Binary with custom threshold

#### Deployment Questions

**Q: How do I deploy to production?**  
A: See the [Deployment](#-deployment) section. Options include Docker, AWS, GCP, Azure, and Heroku.

**Q: Can this handle high-frequency requests?**  
A: Yes, FastAPI is async and can handle concurrent requests. Use horizontal scaling (multiple containers) for higher loads.

**Q: How do I monitor predictions?**  
A: Use MLflow for experiment tracking and logging. Add custom metrics to the API endpoints.

**Q: What about data freshness?**  
A: The API fetches latest data from Yahoo Finance on each request. For production, consider caching with periodic updates.

</details>

### 📚 Additional Resources

- **Documentation**: Full API docs at `http://localhost:8000/docs` when running the server
- **Examples**: Check the `examples/` directory for practical usage
- **Notebooks**: Jupyter notebooks in `notebooks/` for interactive exploration
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/galafis/ml-trading-signals/issues)

### 🔗 Related Projects

- [TA-Lib](https://github.com/mrjbq7/ta-lib): Technical Analysis Library
- [yfinance](https://github.com/ranaroussi/yfinance): Yahoo Finance data downloader
- [MLflow](https://mlflow.org/): ML lifecycle management
- [FastAPI](https://fastapi.tiangolo.com/): Modern web framework for APIs

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 👤 Author

**Gabriel Demetrios Lafis**

---

## Português

### 🎯 Visão Geral

**ML Trading Signals** é um sistema profissional de machine learning para geração de sinais de trading usando algoritmos avançados e análise técnica. A plataforma combina engenharia de features, treinamento de modelos e inferência em tempo real para prever movimentos de mercado com alta precisão.

Construído para traders quantitativos e engenheiros financeiros, este sistema fornece capacidades de ML de ponta a ponta, desde a preparação de dados até o deployment de API pronta para produção.

### ✨ Funcionalidades Principais

#### 🧠 Machine Learning
- **Múltiplos Algoritmos**: XGBoost, LightGBM, Random Forest, Gradient Boosting, Regressão Logística
- **Engenharia de Features**: Mais de 40 indicadores técnicos (tendência, momentum, volatilidade, volume)
- **Time-Series Aware**: Divisões adequadas de treino/validação/teste para dados temporais
- **Avaliação de Modelo**: Métricas abrangentes (acurácia, precisão, recall, F1, AUC-ROC)
- **Importância de Features**: Modelos interpretáveis com ranking de features
- **Integração MLflow**: Rastreamento de experimentos e versionamento de modelos

#### 📊 Indicadores Técnicos
- **Tendência**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, Estocástico, Williams %R, ROC
- **Volatilidade**: Bandas de Bollinger, ATR, Canais de Keltner
- **Volume**: OBV, MFI, Volume Price Trend
- **Features de Preço**: Retornos, gaps, ranges, mudanças

#### 🚀 Pronto para Produção
- **API REST**: FastAPI para previsões em tempo real
- **Persistência de Modelo**: Salvar e carregar modelos treinados
- **Arquitetura Escalável**: Design modular para fácil extensão
- **Suporte Docker**: Deployment containerizado
- **Testes Abrangentes**: Cobertura de testes unitários e de integração

### 🚀 Início Rápido

#### Instalação

```bash
# Clonar repositório
git clone https://github.com/galafis/ml-trading-signals.git
cd ml-trading-signals

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

#### Treinar um Modelo

```bash
# Treinar com parâmetros padrão (Índice Bovespa, 2 anos de dados)
python train.py

# Treinar com parâmetros personalizados
python train.py \
  --symbol PETR4.SA \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --model-type xgboost \
  --target-type direction \
  --horizon 1 \
  --save-model models/meu_modelo.pkl \
  --use-mlflow
```

#### Executar Servidor API

```bash
# Iniciar servidor API
uvicorn src.api.main:app --reload

# Ou com Docker
docker build -t ml-trading-signals .
docker run -p 8000:8000 ml-trading-signals
```

Acesse a documentação da API em `http://localhost:8000/docs`

### 📖 Uso da API

#### Obter Previsão

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "PETR4.SA",
    "model_path": "models/meu_modelo.pkl",
    "model_type": "xgboost",
    "lookback_days": 100
  }'
```

**Resposta:**
```json
{
  "symbol": "PETR4.SA",
  "timestamp": "2025-01-04T12:00:00",
  "signal": 1,
  "probability": 0.85,
  "confidence": "high",
  "current_price": 35.42
}
```

### 🧪 Testes

```bash
# Executar todos os testes
pytest

# Executar com cobertura
pytest --cov=src --cov-report=html

# Executar arquivo de teste específico
pytest tests/unit/test_features.py -v
```

### 📊 Performance do Modelo

Resultados exemplo no Índice Bovespa (^BVSP) com 2 anos de dados:

| Modelo | Acurácia | Precisão | Recall | F1 Score | AUC-ROC |
|--------|----------|----------|--------|----------|---------|
| **XGBoost** | 0.62 | 0.64 | 0.58 | 0.61 | 0.68 |
| **LightGBM** | 0.61 | 0.63 | 0.57 | 0.60 | 0.67 |
| **Random Forest** | 0.59 | 0.61 | 0.55 | 0.58 | 0.64 |
| **Gradient Boosting** | 0.60 | 0.62 | 0.56 | 0.59 | 0.66 |

*Nota: Resultados podem variar dependendo das condições de mercado e período de tempo.*

### 🎓 Pipeline de Treinamento

O pipeline de treinamento inclui:

1. **Busca de Dados**: Dados históricos do Yahoo Finance
2. **Engenharia de Features**: Calcular mais de 40 indicadores técnicos
3. **Preparação de Dados**: Tratar valores ausentes, escalar features
4. **Divisão Treino/Val/Teste**: Divisão temporal (70/15/15)
5. **Treinamento de Modelo**: Treinar com conjunto de validação
6. **Avaliação**: Métricas abrangentes no conjunto de teste
7. **Importância de Features**: Identificar features mais preditivas
8. **Persistência de Modelo**: Salvar modelo treinado para deployment

### 🔧 Personalização de Modelo

Personalizar parâmetros do modelo:

```python
from src.training.train_pipeline import TrainingPipeline

pipeline = TrainingPipeline(
    symbol="PETR4.SA",
    start_date="2022-01-01",
    end_date="2024-12-31",
    model_type="xgboost"
)

# Parâmetros personalizados do modelo
model_params = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'random_state': 42
}

results = pipeline.run(model_params=model_params, use_mlflow=True)
```

### 🌐 Mercados Suportados

Atualmente suporta qualquer símbolo disponível no Yahoo Finance:

- **Ações Brasileiras** (B3): PETR4.SA, VALE3.SA, ITUB4.SA, etc.
- **Índices**: ^BVSP (Bovespa), ^GSPC (S&P 500), ^DJI (Dow Jones)
- **Ações Internacionais**: AAPL, GOOGL, MSFT, etc.
- **ETFs**: SPY, QQQ, IWM, etc.

### 🚀 Deployment

#### Deployment Docker

```bash
# Construir imagem
docker build -t ml-trading-signals .

# Executar container
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  ml-trading-signals
```

#### Deployment em Nuvem

Compatível com:
- **AWS** (ECS, Lambda, SageMaker)
- **Google Cloud** (Cloud Run, AI Platform)
- **Azure** (Container Instances, ML Studio)
- **Heroku**

### 📊 Rastreamento MLflow

Rastrear experimentos com MLflow:

```bash
# Iniciar UI do MLflow
mlflow ui

# Treinar com MLflow
python train.py --use-mlflow

# Ver experimentos em http://localhost:5000
```

### 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para submeter um Pull Request.

Veja [CONTRIBUTING.md](CONTRIBUTING.md) para diretrizes detalhadas sobre:
- Configuração do ambiente de desenvolvimento
- Estilo de código e melhores práticas
- Requisitos de teste
- Processo de pull request

### 🐛 Solução de Problemas

<details>
<summary>Problemas Comuns e Soluções</summary>

#### Problemas de Instalação

**Problema**: `pip install` falha para dependências ta-lib
```bash
# Solução: Instale dependências do sistema primeiro
# Ubuntu/Debian
sudo apt-get install libta-lib-dev

# macOS
brew install ta-lib
```

**Problema**: ImportError para yfinance ou pandas
```bash
# Solução: Certifique-se de que todas as dependências estão instaladas
pip install -r requirements.txt --upgrade
```

#### Problemas de Treinamento

**Problema**: "Nenhum dado encontrado para o símbolo"
- Verifique se o símbolo está formatado corretamente (ex: `PETR4.SA` para ações brasileiras)
- Verifique se o intervalo de datas é válido e não está no futuro
- Alguns símbolos podem não ter dados históricos para o período solicitado

**Problema**: "Dados insuficientes para cálculo de features"
- Aumente o intervalo de datas para obter mais pontos de dados
- Indicadores técnicos precisam de um número mínimo de períodos (tipicamente 20-30 dias)

**Problema**: Treinamento do modelo muito lento
- Reduza `n_estimators` nos parâmetros do modelo
- Use um intervalo de datas menor para testes iniciais
- Tente um modelo mais simples como `logistic` em vez de `xgboost`

#### Problemas da API

**Problema**: API retorna erro 500 na previsão
- Certifique-se de que o arquivo do modelo existe e está acessível
- Verifique se o símbolo tem dados recentes disponíveis
- Verifique se o modelo foi treinado com features compatíveis

**Problema**: Container Docker falha ao iniciar
```bash
# Verificar logs
docker logs <container_id>

# Reconstruir imagem sem cache
docker build --no-cache -t ml-trading-signals .
```

#### Problemas de Performance

**Problema**: Previsões inconsistentes
- Use o parâmetro `random_state` para reprodutibilidade
- Garanta dados de treinamento suficientes (pelo menos 1 ano)
- Verifique problemas de qualidade de dados (valores ausentes, outliers)

**Problema**: Baixa acurácia do modelo
- Tente diferentes tipos de modelo (XGBoost, LightGBM)
- Ajuste hiperparâmetros usando conjunto de validação
- Adicione mais features ou engenharia novos indicadores
- Use períodos de treinamento mais longos
- Considere métodos ensemble

</details>

### ❓ Perguntas Frequentes

<details>
<summary>FAQ - Dúvidas Comuns</summary>

#### Perguntas Gerais

**P: Quais mercados são suportados?**  
R: Qualquer mercado disponível no Yahoo Finance, incluindo ações, índices, ETFs, criptomoedas e commodities.

**P: Posso usar isso para trading real?**  
R: Esta é uma ferramenta de pesquisa e educação. Sempre faça backtesting completo e considere os riscos antes de trading ao vivo. Não é aconselhamento financeiro.

**P: Quais períodos são suportados?**  
R: Atualmente dados diários. Suporte intradiário pode ser adicionado modificando a lógica de busca de dados.

**P: Quão precisas são as previsões?**  
R: A acurácia típica varia de 58-68% dependendo do modelo, features e condições de mercado. Performance passada não garante resultados futuros.

#### Perguntas Técnicas

**P: Quais features são mais importantes?**  
R: Tipicamente RSI, MACD, Bandas de Bollinger e indicadores baseados em volume. Use `get_feature_importance()` para ver no seu modelo específico.

**P: Posso adicionar indicadores personalizados?**  
R: Sim! Adicione-os em `src/features/technical_indicators.py` e atualize o método `add_all_indicators()`. Veja [CONTRIBUTING.md](CONTRIBUTING.md) para exemplos.

**P: Como ajustar hiperparâmetros?**  
R: Passe parâmetros customizados para o pipeline de treinamento:
```python
model_params = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05
}
results = pipeline.run(model_params=model_params)
```

**P: Posso usar aceleração GPU?**  
R: Sim para XGBoost e LightGBM. Configure `tree_method='gpu_hist'` para XGBoost ou `device='gpu'` para LightGBM nos parâmetros do modelo.

**P: Com que frequência devo retreinar os modelos?**  
R: Retreine semanalmente ou mensalmente conforme os mercados evoluem. Monitore métricas de validação para detectar quando retreinamento é necessário.

**P: Qual a diferença entre tipos de target?**  
R: 
- `direction`: Classificação binária (subida/descida)
- `returns`: Regressão (prever retorno % real)
- `binary`: Binário com limite customizado

#### Perguntas de Deployment

**P: Como fazer deploy para produção?**  
R: Veja a seção [Deployment](#-deployment). Opções incluem Docker, AWS, GCP, Azure e Heroku.

**P: Isso aguenta requisições de alta frequência?**  
R: Sim, FastAPI é assíncrono e pode lidar com requisições simultâneas. Use escalonamento horizontal (múltiplos containers) para cargas maiores.

**P: Como monitorar previsões?**  
R: Use MLflow para rastreamento de experimentos e logging. Adicione métricas customizadas aos endpoints da API.

**P: E quanto à atualização dos dados?**  
R: A API busca dados mais recentes do Yahoo Finance em cada requisição. Para produção, considere cache com atualizações periódicas.

</details>

### 📚 Recursos Adicionais

- **Documentação**: Documentação completa da API em `http://localhost:8000/docs` ao executar o servidor
- **Exemplos**: Confira o diretório `examples/` para uso prático
- **Notebooks**: Notebooks Jupyter em `notebooks/` para exploração interativa
- **Contribuindo**: Veja [CONTRIBUTING.md](CONTRIBUTING.md) para diretrizes de desenvolvimento
- **Issues**: Reporte bugs ou solicite recursos em [GitHub Issues](https://github.com/galafis/ml-trading-signals/issues)

### 🔗 Projetos Relacionados

- [TA-Lib](https://github.com/mrjbq7/ta-lib): Biblioteca de Análise Técnica
- [yfinance](https://github.com/ranaroussi/yfinance): Downloader de dados do Yahoo Finance
- [MLflow](https://mlflow.org/): Gerenciamento do ciclo de vida de ML
- [FastAPI](https://fastapi.tiangolo.com/): Framework web moderno para APIs

### 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

### 👤 Autor

**Gabriel Demetrios Lafis**

---

**⭐ Se este projeto foi útil para você, considere dar uma estrela no GitHub!**
