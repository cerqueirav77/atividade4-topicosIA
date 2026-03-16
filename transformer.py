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

# Encoder Block

class EncoderBlock:
    def __init__(self):
        self.W_query = np.random.randn(D_MODEL, D_MODEL) * 0.01
        self.W_key   = np.random.randn(D_MODEL, D_MODEL) * 0.01
        self.W_value = np.random.randn(D_MODEL, D_MODEL) * 0.01
        self.W1 = np.random.randn(D_MODEL, D_FF)   * 0.01
        self.b1 = np.zeros((1, 1, D_FF))
        self.W2 = np.random.randn(D_FF, D_MODEL)   * 0.01
        self.b2 = np.zeros((1, 1, D_MODEL))

    def forward(self, X: np.ndarray) -> np.ndarray:
        Q = X @ self.W_query
        K = X @ self.W_key
        V = X @ self.W_value

        saida_atencao = scaled_dot_product_attention(Q, K, V)
        X_norm1 = add_and_norm(X, saida_atencao)

        saida_ffn = feed_forward_network(X_norm1, self.W1, self.b1, self.W2, self.b2)
        X_out = add_and_norm(X_norm1, saida_ffn)

        return X_out


encoder_stack = [EncoderBlock() for _ in range(N_CAMADAS)]

SEQ_LEN_ENC = 5
X_teste = np.random.randn(BATCH_SIZE, SEQ_LEN_ENC, D_MODEL)
saida_encoder = X_teste
for bloco in encoder_stack:
    saida_encoder = bloco.forward(saida_encoder)

Z = saida_encoder
print(f"\nEncoder Stack — shape de entrada : {X_teste.shape}")
print(f"Encoder Stack — shape de saída Z : {Z.shape}")
print(f"Validação shapes iguais? {X_teste.shape == Z.shape}")