import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ==============================================
# Treinamento de um VAE no dataset PneumoniaMNIST
# ==============================================
# Este script: 
# - Cria/treina um VAE (encoder + sampler + decoder) em imagens 28x28x1
# - Usa medmnist.PneumoniaMNIST para carregar os dados
# - Normaliza para [0,1] e garante shape (28,28,1)
# - Define perda de reconstrução + divergência KL
# - Valida no conjunto de validação
# - Salva pesos do modelo e figura de reconstruções
# ==============================================

# Hiperparâmetros principais
LATENT_DIM = 16
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3

# Diretórios de saída
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
WEIGHTS_PATH = os.path.join(MODELS_DIR, 'vae_pneumonia.weights.h5')
CONFIG_PATH = os.path.join(MODELS_DIR, 'config.json')
RECON_FIG_PATH = os.path.join(OUTPUTS_DIR, 'reconstructions.png')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Fixar seeds para reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

# Opcional: configurar memória da GPU (se houver)
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass


def load_pneumonia_mnist(split: str):
    """Carrega imagens do PneumoniaMNIST em numpy com shape (N, 28, 28, 1) e valores em [0,1].

    split: 'train', 'val' ou 'test'
    """
    from medmnist import PneumoniaMNIST  # import tardio para evitar custo quando não usado

    dataset = PneumoniaMNIST(split=split, download=True)
    # dataset.imgs tipicamente é uint8 no shape (N,28,28,1) para grayscale
    images: np.ndarray = dataset.imgs

    # Garantir float32 e normalização [0,1]
    images = images.astype('float32')
    if images.max() > 1.0:
        images = images / 255.0

    # Garantir shape (N,28,28,1)
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)

    return images


def create_tf_dataset(images: np.ndarray, batch_size: int, training: bool) -> tf.data.Dataset:
    """Cria um tf.data.Dataset a partir de um array de imagens normalizadas."""
    ds = tf.data.Dataset.from_tensor_slices(images)
    if training:
        ds = ds.shuffle(buffer_size=min(len(images), 10_000), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class Sampling(tf.keras.layers.Layer):
    """Camada de reparametrização: z = mean + exp(0.5 * log_var) * epsilon"""
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(latent_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')


def build_decoder(latent_dim: int) -> tf.keras.Model:
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    outputs = tf.keras.layers.Conv2DTranspose(1, kernel_size=3, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(latent_inputs, outputs, name='decoder')


class VAE(tf.keras.Model):
    """Variational Autoencoder com perda composta (reconstrução + KL)."""
    def __init__(self, encoder: tf.keras.Model, decoder: tf.keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.recon_loss_tracker = tf.keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = tf.keras.metrics.Mean(name='kl_loss')

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)
        return reconstruction

    def compute_losses(self, data, reconstruction, z_mean, z_log_var):
        # Perda de reconstrução (BCE por pixel). A BCE do Keras reduz o último eixo,
        # então o tensor resultante pode ser (batch, 28, 28). Somamos todas as
        # dimensões após o batch, de forma robusta para 3D ou 4D.
        bce = tf.keras.losses.binary_crossentropy(data, reconstruction)
        axes = tf.range(1, tf.rank(bce))
        recon_per_example = tf.reduce_sum(bce, axis=axes)
        recon_loss = tf.reduce_mean(recon_per_example)
        # Divergência KL (média no batch)
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)
            total_loss, recon_loss, kl_loss = self.compute_losses(data, reconstruction, z_mean, z_log_var)
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)
        total_loss, recon_loss, kl_loss = self.compute_losses(data, reconstruction, z_mean, z_log_var)
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}


def visualize_reconstructions(model: VAE, images: np.ndarray, save_path: str, num_examples: int = 8):
    """Gera uma figura comparando originais vs reconstruções e salva em arquivo."""
    assert images.ndim == 4 and images.shape[1:] == (28, 28, 1)
    num_examples = min(num_examples, images.shape[0])
    batch = images[:num_examples]
    recons = model.predict(batch, verbose=0)

    plt.figure(figsize=(2 * num_examples, 4))
    for i in range(num_examples):
        # Linha 1: originais
        ax1 = plt.subplot(2, num_examples, i + 1)
        plt.imshow(batch[i].squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0:
            ax1.set_title('Original')
        # Linha 2: reconstruções
        ax2 = plt.subplot(2, num_examples, num_examples + i + 1)
        plt.imshow(recons[i].squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0:
            ax2.set_title('Reconstruída')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    # 1) Carregar dados
    print('Carregando PneumoniaMNIST...')
    x_train = load_pneumonia_mnist(split='train')
    x_val = load_pneumonia_mnist(split='val')

    print(f"Treino: {x_train.shape}, Validação: {x_val.shape}")

    # 2) Datasets
    ds_train = create_tf_dataset(x_train, BATCH_SIZE, training=True)
    ds_val = create_tf_dataset(x_val, BATCH_SIZE, training=False)

    # 3) Modelo
    encoder = build_encoder(LATENT_DIM)
    decoder = build_decoder(LATENT_DIM)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    # Build the model once so that saving weights works (subclassed model)
    _ = vae(tf.zeros((1, 28, 28, 1)), training=False)

    # 4) Treinamento
    print('Iniciando treinamento...')
    vae.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        verbose=2
    )

    # 5) Salvar pesos e configuração
    print(f'Salvando pesos em: {WEIGHTS_PATH}')
    vae.save_weights(WEIGHTS_PATH)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump({'latent_dim': LATENT_DIM}, f, ensure_ascii=False, indent=2)

    # 6) Visualização de reconstruções
    print(f'Gerando figura de reconstruções em: {RECON_FIG_PATH}')
    visualize_reconstructions(vae, x_val, save_path=RECON_FIG_PATH, num_examples=8)

    print('Concluído.')


if __name__ == '__main__':
    main() 