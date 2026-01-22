# INSTRUÇÕES DE USO — PROJETO CLASSIFICADOR DE OUTFIT

Autores: Leonardo Ferreira e Pedro Arthur da Silva
Projeto: Classificação de Estilos de Moda com Visão Computacional
Tecnologias: Python, TensorFlow, OpenCV, Streamlit


1. VISÃO GERAL DO PROJETO
------------------------
Este projeto implementa um sistema de classificação automática de estilos
de moda a partir de imagens de outfits completos.

O sistema recebe uma imagem enviada pelo usuário e retorna:
- Estilo principal (Macro-classe)
- Subestilo (Subclasse coerente)
- Evidências semânticas (tags mais frequentes no dataset)

A aplicação foi desenvolvida para demonstração interativa via navegador web.


2. REQUISITOS DO SISTEMA
-----------------------
- Python 3.9 ou superior
- Sistema operacional: Windows / Linux / macOS
- Mínimo recomendado:
  • 8 GB de RAM
  • CPU (GPU não é obrigatória)

Bibliotecas principais:
- tensorflow
- streamlit
- opencv-python
- pillow
- pandas
- numpy


3. ESTRUTURA DO PROJETO
----------------------
A estrutura esperada do projeto é:

/digital-image-processing-fashion-trends
│/fashion_trends
├── app.py                    → Aplicação web Streamlit
├── INSTRUCOES.txt            → Este arquivo
│
├── model/
│   ├── model.keras           → Modelo treinado
│   └── labels_runtime.json   → Nomes das classes e parâmetros
│
├── data/
│   └── splits.csv            → Dataset processado + metadata
│
└── venv/                     → Ambiente virtual (opcional)


4. CRIAÇÃO DO AMBIENTE VIRTUAL (RECOMENDADO)
--------------------------------------------
No terminal, dentro da pasta do projeto:

Windows:
python -m venv venv
venv\Scripts\activate

Linux / macOS:
python3 -m venv venv
source venv/bin/activate


5. INSTALAÇÃO DAS DEPENDÊNCIAS
------------------------------
Com o ambiente ativado, execute:

pip install tensorflow streamlit opencv-python pillow pandas numpy


6. EXECUTANDO A APLICAÇÃO
-------------------------
No diretório do projeto, execute:

streamlit run app.py

O terminal exibirá algo como:
Local URL: http://localhost:8501

Abra esse endereço em qualquer navegador.


7. UTILIZAÇÃO DA APLICAÇÃO
--------------------------
1. Acesse a interface web
2. Faça upload de uma imagem de outfit completo
   Formatos aceitos:
   - JPG / JPEG
   - PNG
   - WEBP
   - JFIF
3. Aguarde a análise
4. O sistema exibirá:
   - Estilo principal (em português)
   - Subestilo (em português)
   - Probabilidades
   - Tags semânticas associadas ao estilo

Observação:
As traduções são apenas para a interface.
Internamente o modelo trabalha com rótulos padronizados em inglês.


8. ESTILOS RECONHECIDOS
----------------------
Macro-classes (nível alto):
- Casual / Básico
- Chique / Romântico
- Moderno / Eclético
- Expressivo / Autoral

Subestilos (nível detalhado):
- Casual, Básico, Confortável, Jeans, Clássico
- Chique, Romântico, Elegante, Social Jovem
- Urbano, Tendência, Eclético
- Boho, Rocker, Sensual


9. EXPLICAÇÃO DO FUNCIONAMENTO INTERNO
--------------------------------------
O sistema utiliza uma rede neural convolucional (CNN) com backbone MobileNetV2.

A arquitetura é multitarefa:
- Uma cabeça prevê a macro-classe
- Outra cabeça prevê a subclasse

A subclasse é escolhida de forma coerente com a macro prevista,
evitando contradições semânticas.

Além da predição visual, o sistema apresenta evidências semânticas
(tags mais frequentes no dataset para aquela classe),
aumentando a interpretabilidade do resultado.
