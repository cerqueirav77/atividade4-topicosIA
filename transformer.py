import numpy as np

D_MODEL    = 512
D_FF       = 2048
VOCAB_SIZE = 10_000
BATCH_SIZE = 1
N_CAMADAS  = 6
EPSILON    = 1e-6

np.random.seed(42)


def softmax(matriz: np.ndarray) -> np.ndarray:
    deslocado = matriz - np.max(matriz, axis=-1, keepdims=True)
    exponenciais = np.exp(deslocado)
    return exponenciais / np.sum(exponenciais, axis=-1, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mascara: np.ndarray = None
) -> np.ndarray:
    dimensao_k = K.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(dimensao_k)

    if mascara is not None:
        scores = scores + mascara

    pesos = softmax(scores)
    return pesos @ V


def layer_norm(tensor: np.ndarray) -> np.ndarray:
    media    = np.mean(tensor, axis=-1, keepdims=True)
    variancia = np.var(tensor, axis=-1, keepdims=True)
    return (tensor - media) / np.sqrt(variancia + EPSILON)


def add_and_norm(tensor_entrada: np.ndarray, saida_sublayer: np.ndarray) -> np.ndarray:
    return layer_norm(tensor_entrada + saida_sublayer)


def feed_forward_network(
    tensor: np.ndarray,
    W1: np.ndarray, b1: np.ndarray,
    W2: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    camada_expandida = np.maximum(0, tensor @ W1 + b1)
    return camada_expandida @ W2 + b2


def create_causal_mask(seq_len: int) -> np.ndarray:
    return np.triu(np.full((seq_len, seq_len), -np.inf), k=1)

print("Blocos base carregados com sucesso.")