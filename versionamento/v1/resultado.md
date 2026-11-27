# Relatório de Desenvolvimento

## Código base:
O código base é _sequencial em CPU_ e possui tempo de execução de **14.5509 s** no Cluster SDumont.

Saída do código base:
---

```raw
TREINO | 3823 imagens
Epoca  1/51 | perda media 0.4129
Epoca  6/51 | perda media 0.0442
Epoca 11/51 | perda media 0.0312
Epoca 16/51 | perda media 0.0261
Epoca 21/51 | perda media 0.0227
Epoca 26/51 | perda media 0.0204
Epoca 31/51 | perda media 0.0184
Epoca 36/51 | perda media 0.0169
Epoca 41/51 | perda media 0.0155
Epoca 46/51 | perda media 0.0145
Epoca 51/51 | perda media 0.0135
Tempo de treino 14.5509 s

AVALIACAO FINAL
Acuracia 95.5% | 1717 acertos em 1797 imagens
Perda media 0.0432

ALGUMAS INFERENCIAS
Esperado  Previsto
       0         0
       1         1
       2         8
       3         3
       5         9
       7         9

CONFUSOES MAIS FREQUENTES
Esperado  Previsto  Ocorrencias
       7         9            9
       8         1            6
       3         2            5
```

## Problema 1

### Evidência
![evidencia](dense_backward.png)

### Primeira Iteração
Foi alterado o `Baren::Backward` para que os loops sejam paralelizados

```raw
Tempo de treino 33.4338 s
```


![evidencia](otim1_profiling.png)
**38% do tempo da GPU foi dedicado à transferência de dados**

HIPOTESE: Aumentando o tamanho dos blocos a transferência fica menos recorrente.
32 threads/bloco: "Tempo de treino 31.1255 s"
128 threads/bloco: "Tempo de treino 33.4338 s"
256 threads/bloco: "Tempo de treino 30.8344 s"
1024 threads/bloco: "Tempo de treino 31.0147 s"

CONCLUSÃO: O tamanho dos blocos não é o culpado pela transferência

### Segunda Iteração
Otimização dos recursos de memória.
Ao invés de chamar o cudaMalloc toda a vez que a função `Backward` for chamada, deixar o setup da GPU apenas no construtor que será lançado na criação do objeto. Assim, sempre que for criado uma nova instância, o código aloca de uma vez o espaço necessário para ela em GPU.

```raw
Tempo de treino 19.0765 s
```

## Resultado
![evidencia](otim1.1_profiling.png)
O resultado final é mais lento que o caso base.
Isso se deve ao fato do `Forward` não utilizar paralelismo de GPU e exigir que vários dados sejam sincronizados da CPU a cada chamada da `Backwards` e não aproveitem da memória compartilhada da GPU.

Com a paralelização do Forward, assim como o Backward, é possível fazer todas as operações diretamente em GPU exigindo poucos carregamentos de volta para a CPU.


Resumo das etapas
---

| Hardware de execução |                  Commit                  | Tempo de execução |
| :------------------: | :--------------------------------------: | :---------------: |
| Monstrao (merry_cpu) |              (Código Base)               |      14.55s       |
| Monstrao (merry_gpu) | Backward Propagation paralelizada em GPU |      19.07s       |
| Monstrao (merry_gpu) | Otimização da alocação da memória de GPU |      33.43s       |