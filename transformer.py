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

# Decoder Block 

class DecoderBlock:


    def __init__(self):
        self.W_query_self  = np.random.randn(D_MODEL, D_MODEL) * 0.01
        self.W_key_self    = np.random.randn(D_MODEL, D_MODEL) * 0.01
        self.W_value_self  = np.random.randn(D_MODEL, D_MODEL) * 0.01

        self.W_query_cross = np.random.randn(D_MODEL, D_MODEL) * 0.01
        self.W_key_cross   = np.random.randn(D_MODEL, D_MODEL) * 0.01
        self.W_value_cross = np.random.randn(D_MODEL, D_MODEL) * 0.01

        self.W1 = np.random.randn(D_MODEL, D_FF)  * 0.01
        self.b1 = np.zeros((1, 1, D_FF))
        self.W2 = np.random.randn(D_FF, D_MODEL)  * 0.01
        self.b2 = np.zeros((1, 1, D_MODEL))

        self.W_proj = np.random.randn(D_MODEL, VOCAB_SIZE) * 0.01

    def forward(self, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:

        seq_len_dec = Y.shape[1]
        mascara = create_causal_mask(seq_len_dec)

        Q_self = Y @ self.W_query_self
        K_self = Y @ self.W_key_self
        V_self = Y @ self.W_value_self
        saida_self = scaled_dot_product_attention(Q_self, K_self, V_self, mascara)
        Y_norm1 = add_and_norm(Y, saida_self)

        Q_cross = Y_norm1 @ self.W_query_cross
        K_cross = Z      @ self.W_key_cross
        V_cross = Z      @ self.W_value_cross
        saida_cross = scaled_dot_product_attention(Q_cross, K_cross, V_cross)
        Y_norm2 = add_and_norm(Y_norm1, saida_cross)

        saida_ffn = feed_forward_network(Y_norm2, self.W1, self.b1, self.W2, self.b2)
        Y_out = add_and_norm(Y_norm2, saida_ffn)

        logits = Y_out @ self.W_proj
        probabilidades = softmax(logits)

        return probabilidades


SEQ_LEN_DEC = 3
Y_teste = np.random.randn(BATCH_SIZE, SEQ_LEN_DEC, D_MODEL)
decoder_block = DecoderBlock()
probs_teste = decoder_block.forward(Y_teste, Z)

print(f"\nDecoder Block — shape entrada Y  : {Y_teste.shape}")
print(f"Decoder Block — shape memória Z  : {Z.shape}")
print(f"Decoder Block — shape saída probs: {probs_teste.shape}")
print(f"Validação soma probs última pos  : {probs_teste[0, -1, :].sum():.6f}")