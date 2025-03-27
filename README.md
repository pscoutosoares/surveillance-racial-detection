# Análise de Viés em Detecção de Faces em Vídeos de Crimes

Este projeto visa analisar e quantificar possíveis vieses na detecção e classificação de faces em vídeos de crimes, utilizando a base de dados UCF-Crimes e o modelo FairFace.

## Estrutura do Projeto

```
.
├── data/               # Dados brutos e processados
│   ├── raw/           # Vídeos originais
│   ├── processed/     # Frames extraídos
│   └── faces/         # Faces detectadas e alinhadas
├── src/               # Código fonte
│   ├── preprocessing/ # Scripts de pré-processamento
│   ├── models/       # Implementação dos modelos
│   ├── evaluation/   # Scripts de avaliação
│   └── utils/        # Funções utilitárias
├── notebooks/        # Jupyter notebooks para análise
├── results/          # Resultados e métricas
└── docs/            # Documentação
```

## Requisitos

- Python 3.8+
- PyTorch
- OpenCV
- RetinaFace
- FairFace
- Outras dependências listadas em `requirements.txt`

## Instalação

1. Clone o repositório
2. Crie um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Uso

1. Pré-processamento dos dados:

```bash
python src/preprocessing/extract_faces.py
```

2. Classificação com FairFace:

```bash
python src/models/classify_faces.py
```

3. Avaliação de viés:

```bash
python src/evaluation/bias_analysis.py
```

## Metodologia

1. Coleta e Pré-processamento

   - Extração de frames dos vídeos
   - Detecção de faces com RetinaFace
   - Alinhamento facial e super-resolução

2. Classificação

   - Aplicação do modelo FairFace
   - Tratamento de casos especiais

3. Avaliação
   - Métricas de performance
   - Análise de viés
   - Testes estatísticos
