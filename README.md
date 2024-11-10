# Detecção de anomalias em um motor elétrico em funcionamento

Neste repositório pretende-se aplicar os algoritmos CNN e Efficient KAN para detecção de anomalias em relação a imagens termográficas coletadas a partir de um motor elétrico em funcionamento.

## CNN

Para o modelo CNN, foi utilizado a arquitetura apresentada a seguir:

- Conv2d;

- BatchNorm2d;

- ReLU;

- Conv2d;

- BatchNorm2d;

- ReLU;

- Conv2d;

- BatchNorm2d;

- ReLU;

- MaxPool2d;

- Flatten;

- Linear;

- Dropout;

- ReLU;

- Linear.

### Treinamento

- Época 1/35 - Perda no treinamento: 0.729908 - Acurácia: 0.4667;

- Época 35/35 - Perda no treinamento: 0.000000 - Acurácia: 1.0000;

- Tempo total de treinamento: 8.41 segundos.

### Teste

- Acurácia: 100.00%;

- Precisão: 100.00%;

- Recall: 100.00%;

- F1 Score: 100.00%.

## Efficient KAN

Para o modelo Efficient KAN, foi utilizado a arquitetura apresentada a seguir:

- Camada de entrada;

- Camada oculta;

- Camada de saída.

### Treinamento

- Época 1/35 - Perda no treinamento: 0.681536 - Acurácia: 0.6667;

- Época 35/35 - Perda no treinamento: 0.000000 - Acurácia: 1.0000;

- Tempo total de treinamento: 6.32 segundos.

### Teste

- Acurácia: 100.00%;

- Precisão: 100.00%;

- Recall: 100.00%;

- F1 Score: 100.00%.