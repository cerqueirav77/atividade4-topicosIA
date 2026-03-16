# Transformer Completo From Scratch — LAB 04

## Descrição

Implementação completa da arquitetura Transformer Encoder-Decoder,
baseada no paper *"Attention Is All You Need"* (Vaswani et al., 2017).

Este laboratório integra todos os blocos matemáticos construídos nos labs anteriores
em uma única topologia coerente, realizando um teste de tradução fim-a-fim com
uma sequência de brinquedo (*toy sequence*).

## Como Rodar

**Pré-requisitos:** Python 3.x e NumPy.

1. Crie e ative um ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Instale as dependências:
```bash
pip install numpy
```

3. Execute o transformer:
```bash
python3 transformer.py
```

4. Desative o ambiente virtual ao terminar (opcional):
```bash
deactivate
```

## Arquitetura Implementada

### Blocos Base (Lab 01, 02 e 03)
- `softmax` — com estabilidade numérica
- `scaled_dot_product_attention(Q, K, V, mask)` — com suporte a máscara opcional
- `layer_norm` — normalização no último eixo
- `add_and_norm` — conexão residual + LayerNorm
- `feed_forward_network` — expansão ReLU de D_MODEL → D_FF → D_MODEL
- `create_causal_mask` — máscara triangular superior com -∞

### EncoderBlock (Tarefa 2)

Cada bloco executa o seguinte fluxo:
```
1. Q, K, V = X @ W_query, X @ W_key, X @ W_value
2. X = Add & Norm(X, SelfAttention(Q, K, V))
3. X = Add & Norm(X, FFN(X))
```

6 blocos são empilhados para produzir a memória contextualizada **Z**.

### DecoderBlock (Tarefa 3)

Cada bloco executa o seguinte fluxo:
```
1. Y = Add & Norm(Y, MaskedSelfAttention(Y, mask))
2. Y = Add & Norm(Y, CrossAttention(Q=Y, K=Z, V=Z))
3. Y = Add & Norm(Y, FFN(Y))
4. probs = Softmax(Y @ W_proj)
```

### Inferência Auto-Regressiva (Tarefa 4)
```
Entrada : "Thinking Machines" → encoder_input (1, 2, 512)
Memória : Z = EncoderStack(encoder_input)   (1, 2, 512)

Loop while:
  probs = DecoderBlock(Y_atual, Z)
  token = vocabulario[argmax(probs)]
  sequencia.append(token)
  if token == <EOS>: break
```

## Exemplo de Output
```
Encoder Stack — shape de entrada : (1, 5, 512)
Encoder Stack — shape de saída Z : (1, 5, 512)
Validação shapes iguais? True

Decoder Block — shape saída probs: (1, 3, 10000)
Validação soma probs última pos  : 1.000000

Entrada simulada: 'Thinking Machines'
Passo 01 — token gerado: 'word_XXXX'
...
Sequência final gerada: <START> word_X word_Y ...
```

## Referência

Vaswani, A. et al. **Attention Is All You Need**, 2017.
https://arxiv.org/abs/1706.03762

## Créditos

Partes complementadas com IA (Claude — Anthropic), revisadas por Victor Cerqueira.