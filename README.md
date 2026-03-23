# Laboratório 5 - Treinamento de Transformer para Tradução

## Objetivo

Este laboratório tem como objetivo implementar o treinamento fim a fim de um modelo Transformer para uma tarefa de tradução automática, utilizando:

- Dataset do Hugging Face  
- Tokenização com tokenizer pré-treinado  
- Treinamento com forward, loss, backward e optimizer.step()  
- Teste de overfitting com geração de tradução  

---

## Estrutura do Projeto

laboratorio5/
│
├── laboratorio5_transformer.py
└── README.md

---

## Requisitos

Instale as dependências com:

pip install torch datasets transformers sentencepiece

---

## Dataset

Foi utilizado o dataset Multi30k, disponível no Hugging Face:

- Tradução de frases do inglês (en) para alemão (de)
- Utilizamos apenas 1000 pares de frases para treinamento rápido

---

## Tokenização

- Tokenizador utilizado: bert-base-multilingual-cased
- Adição de tokens especiais:
  - <START> → início da sequência do decoder
  - <EOS> → fim da sequência
- Uso de padding para padronizar os batches

---

## Arquitetura do Modelo

O modelo segue a estrutura clássica do Transformer:

- d_model = 128
- nhead = 4
- num_encoder_layers = 2
- num_decoder_layers = 2
- dim_feedforward = 512
- dropout = 0.1

Componentes principais:

- Embeddings (encoder e decoder)
- Positional Encoding
- nn.Transformer (PyTorch)
- Camada linear de saída (Linear)

---

## Treinamento

Configuração do treino:

- Otimizador: Adam
- Learning rate: 1e-4
- Função de perda: CrossEntropyLoss
  - ignore_index = PAD_TOKEN
- Batch size: 32
- Epochs: 10

### Estratégia

- Uso de teacher forcing:
  - Entrada do decoder → sequência deslocada à direita
  - Target → sequência deslocada à esquerda
- Loss calculada ignorando padding

---

## Exemplo de saída do treinamento

Epoch 1/10 - Loss: 5.4321  
Epoch 2/10 - Loss: 4.2103  
...  
Epoch 10/10 - Loss: 2.1345  

---

## Geração de Tradução (Teste de Overfitting)

Após o treinamento, o modelo realiza geração auto-regressiva:

- Entrada: frase do conjunto de treino
- Saída: tradução gerada token por token

### Exemplo:

Frase origem (en): A man is playing guitar  
Tradução real (de): Ein Mann spielt Gitarre  
Tradução gerada: Ein Mann spielt Gitarre  

Esse teste verifica se o modelo conseguiu memorizar o dataset, como esperado em um cenário de overfitting.

---

## Funcionamento do Decode

- Começa com <START>
- A cada passo:
  - Prediz o próximo token
  - Adiciona à sequência
- Para quando encontra <EOS>

---
