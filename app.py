# ==============================================
# DEPENDÊNCIAS — instale com pip antes de executar:
#
#   pip install streamlit tensorflow numpy pandas plotly altair Pillow
#
# Para executar:
#   streamlit run app.py
# ==============================================

# --- Biblioteca padrão ---
import os
import json
import io
import time
import base64
from datetime import datetime

# --- Computação numérica e dados ---
import numpy as np
import pandas as pd

# --- Interface web ---
import streamlit as st

# --- Deep learning ---
import tensorflow as tf

# --- Visualização ---
import plotly.express as px
import altair as alt

# --- Imagens ---
from PIL import Image


# ==============================================
# CAMINHOS
# ==============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
WEIGHTS_PATH = os.path.join(MODELS_DIR, 'vae_pneumonia.weights.h5')
CONFIG_PATH = os.path.join(MODELS_DIR, 'config.json')


# ==============================================
# CAMADA CUSTOMIZADA: Sampling
# ==============================================
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ==============================================
# ENCODER
# ==============================================
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


# ==============================================
# DECODER
# ==============================================
def build_decoder(latent_dim: int) -> tf.keras.Model:
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    outputs = tf.keras.layers.Conv2DTranspose(1, kernel_size=3, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(latent_inputs, outputs, name='decoder')


# ==============================================
# MODELO VAE
# ==============================================
class VAE(tf.keras.Model):
    def __init__(self, encoder: tf.keras.Model, decoder: tf.keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)
        return reconstruction

    def encode(self, inputs, training=False):
        return self.encoder(inputs, training=training)

    def decode(self, z, training=False):
        return self.decoder(z, training=training)


# ==============================================
# CACHE: CARREGAMENTO DO MODELO
# ==============================================
@st.cache_resource
def load_model():
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(WEIGHTS_PATH):
        return None, 'Pesos ou configuração não encontrados. Treine o modelo executando train_vae.py.'

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    latent_dim = int(config.get('latent_dim', 16))

    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    vae = VAE(encoder, decoder)

    dummy = tf.zeros((1, 28, 28, 1))
    _ = vae(dummy, training=False)

    vae.load_weights(WEIGHTS_PATH)
    return vae, None


# ==============================================
# PRÉ-PROCESSAMENTO
# ==============================================
def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != 'L':
        image = image.convert('L')

    if image.size != (28, 28):
        image = image.resize((28, 28))

    arr = np.array(image).astype('float32')
    if arr.max() > 1.0:
        arr = arr / 255.0

    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr


# ==============================================
# CÁLCULOS
# ==============================================
@st.cache_data
def compute_reconstruction_error(x: np.ndarray, x_recon: np.ndarray) -> float:
    return float(np.mean((x - x_recon) ** 2))


@st.cache_data
def classify_pneumonia(reconstruction_error: float, threshold_normal: float, threshold_borderline: float) -> tuple:
    if reconstruction_error < threshold_normal:
        return "NORMAL", "Baixo risco de pneumonia", "green"
    elif reconstruction_error < threshold_borderline:
        return "BORDERLINE", "Risco moderado - recomenda-se avaliação médica", "orange"
    else:
        return "POSSÍVEL PNEUMONIA", "Alto risco - urgente avaliação médica", "red"


def generate_new_images(vae: VAE, num_images: int = 4) -> np.ndarray:
    latent_dim = vae.encoder.output_shape[0][-1]
    z_samples = np.random.normal(0, 1, (num_images, latent_dim))
    generated_images = vae.decode(z_samples, training=False).numpy()
    return generated_images


# ==============================================
# FUNÇÕES AUXILIARES DE UX / ESTADO
# ==============================================
def get_confidence_band(confidence: int) -> tuple:
    if confidence > 85:
        return (
            "Alta confiança",
            "success",
            "O modelo encontrou um padrão mais consistente com o que aprendeu. Ainda assim, este resultado é apenas uma estimativa."
        )
    elif confidence >= 60:
        return (
            "Média confiança",
            "warning",
            "O resultado é intermediário. Recomenda-se interpretação cuidadosa e validação humana."
        )
    else:
        return (
            "Baixa confiança",
            "error",
            "O sistema está incerto sobre este resultado. Recomenda-se revisão humana da imagem."
        )


def reset_analysis():
    st.session_state.analysis_ran = False
    st.session_state.last_result = None
    st.toast("Configuração alterada. Execute novamente a triagem.")


def reset_session():
    keys_to_reset = [
        "history",
        "last_result",
        "feedback_log",
        "analysis_ran",
        "generated_images",
        "history_df",
        "filtered_history",
        "last_file_key",
        "run_file_key",
    ]

    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

    st.toast("Sessão resetada com sucesso.")


def image_to_base64_uri(arr: np.ndarray) -> str:
    """Converte um array numpy (28,28) ou (28,28,1) em data URI base64 para exibição no DataFrame."""
    img = arr.squeeze()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    pil_img = Image.fromarray(img, mode='L')
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def build_feedback_df(feedback_log: list) -> pd.DataFrame:
    if not feedback_log:
        return pd.DataFrame(columns=["Execução", "Imagem", "Classificação", "Erro MSE", "Confiança (%)", "Feedback", "Data/Hora"])

    return pd.DataFrame(feedback_log)


def build_class_distribution_df(history: list) -> pd.DataFrame:
    if not history:
        return pd.DataFrame(columns=["Classificação", "Quantidade"])

    df = pd.DataFrame(history)
    dist = df["classification"].value_counts().reset_index()
    dist.columns = ["Classificação", "Quantidade"]
    return dist


# ==============================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================
st.set_page_config(
    page_title='VAE PneumoniaMNIST - Triagem e Geração',
    layout='wide'
)


# ==============================================
# INICIALIZAÇÃO DE ESTADO
# ==============================================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []

if "analysis_ran" not in st.session_state:
    st.session_state.analysis_ran = False

if "generated_images" not in st.session_state:
    st.session_state.generated_images = None

if "num_generated" not in st.session_state:
    st.session_state.num_generated = 4

if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame(
        columns=["Execução", "Imagem", "Classificação", "Erro MSE", "Confiança (%)", "Data/Hora"]
    )


# ==============================================
# SIDEBAR
# ==============================================
st.sidebar.header("Modelo VAE")

vae, err = load_model()
if err:
    st.sidebar.error(err)
    st.stop()
else:
    st.sidebar.success("✅ Modelo carregado com sucesso!")
    st.sidebar.info(f"Dimensão latente: {vae.encoder.output_shape[0][-1]}")

st.sidebar.markdown("---")
st.sidebar.header("Configurações de Triagem")

st.sidebar.slider(
    "Threshold Normal (MSE)",
    min_value=0.000, max_value=0.050, value=0.010, step=0.001,
    format="%.3f",
    key="threshold_normal",
    on_change=reset_analysis,
    help="MSE abaixo deste valor → NORMAL",
)

st.sidebar.slider(
    "Threshold Borderline (MSE)",
    min_value=0.000, max_value=0.100, value=0.020, step=0.001,
    format="%.3f",
    key="threshold_borderline",
    on_change=reset_analysis,
    help="MSE entre o threshold normal e este valor → BORDERLINE. Acima dele → POSSÍVEL PNEUMONIA",
)

st.sidebar.checkbox(
    "Simular latência",
    value=True,
    key="simulate_latency",
)

st.sidebar.markdown("---")
st.sidebar.header("Geração de Imagens")

st.sidebar.slider(
    "Quantidade de imagens sintéticas",
    min_value=1,
    max_value=8,
    value=4,
    step=1,
    key="num_generated",
    help="Define quantas imagens o VAE irá gerar a partir do espaço latente."
)

st.sidebar.markdown("---")

col_sb1, col_sb2 = st.sidebar.columns(2)

with col_sb1:
    if st.button("Limpar Cache"):
        st.cache_data.clear()
        st.toast("Cache de dados limpo.")

with col_sb2:
    if st.button("Resetar Sessão"):
        reset_session()
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Sobre")
st.sidebar.info(
    "Triagem de pneumonia via VAE usando erro de reconstrução como sinal de anomalia. "
    "Este sistema é educacional e não substitui avaliação médica."
)


# ==============================================
# HEADER PRINCIPAL
# ==============================================
st.title("VAE PneumoniaMNIST — Triagem de Pneumonia e Geração de Imagens")
st.markdown(
    """
    Sistema interativo para **triagem assistida por IA** com base no erro de reconstrução de um
    **Variational Autoencoder (VAE)** treinado no dataset **PneumoniaMNIST**.
    """
)
st.caption(
    "⚠️ Resultado estimado para fins educacionais. Não utilize este sistema como diagnóstico médico definitivo."
)

st.markdown("---")


# ==============================================
# ENTRADA PRINCIPAL
# ==============================================
uploaded = st.file_uploader(
    "Envie uma imagem de raio-X para análise (PNG/JPG)",
    type=["png", "jpg", "jpeg"],
)

col_action1, col_action2 = st.columns([1, 3])

with col_action1:
    if st.button("🔍 Executar Triagem", use_container_width=True):
        if uploaded is None:
            st.warning("Envie uma imagem antes de executar a triagem.")
        else:
            st.session_state.analysis_ran = True
            st.session_state.run_file_key = uploaded.name + str(uploaded.size)

with col_action2:
    st.info("Configure os parâmetros na barra lateral e clique em **🔍 Executar Triagem**.")

if not uploaded:
    st.info("Envie uma imagem de raio-X para iniciar a análise.")
    st.markdown("---")
    st.caption(
        "🩺 **Fluxo sugerido:** enviar imagem → ajustar thresholds → executar triagem → revisar confiança → registrar feedback."
    )
    st.stop()


# ==============================================
# EXECUÇÃO CONTROLADA PELO ESTADO
# ==============================================
if st.session_state.analysis_ran:

    file_key = st.session_state.get("run_file_key", "")

    if st.session_state.get("last_file_key") != file_key:
        if st.session_state.simulate_latency:
            with st.status("Executando pipeline de análise...", expanded=True) as status:
                st.write("1. Carregando imagem enviada pelo usuário...")
                time.sleep(0.4)

                st.write("2. Pré-processando imagem para 28x28 em escala de cinza...")
                time.sleep(0.4)

                st.write("3. Codificando imagem no espaço latente...")
                time.sleep(0.4)

                st.write("4. Reconstruindo imagem com o decoder do VAE...")
                bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    bar.progress(i + 1)

                st.write("5. Calculando erro de reconstrução (MSE)...")
                time.sleep(0.3)

                st.write("6. Classificando o risco com base nos thresholds configurados...")
                time.sleep(0.3)

                status.update(label="Pipeline concluído com sucesso.", state="complete", expanded=False)

            st.toast("Análise concluída com sucesso.")

        st.session_state.last_file_key = file_key

    image = Image.open(io.BytesIO(uploaded.read()))
    x = preprocess_image(image)
    recon = vae(x, training=False).numpy()
    mse = compute_reconstruction_error(x, recon)

    classification, description, color = classify_pneumonia(
        mse,
        st.session_state.threshold_normal,
        st.session_state.threshold_borderline,
    )

    confidence_percent = max(0, int((1 - mse) * 100)) if mse < 1 else 0
    confidence_label, confidence_type, confidence_message = get_confidence_band(confidence_percent)

    if st.session_state.last_result is None or st.session_state.last_result.get("file_key") != file_key:
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        st.session_state.last_result = {
            "x": x,
            "recon": recon,
            "mse": mse,
            "classification": classification,
            "confidence": confidence_percent,
            "confidence_label": confidence_label,
            "description": description,
            "file_key": file_key,
            "timestamp": current_time,
        }

        img_uri = image_to_base64_uri(x[0])

        new_row = pd.DataFrame([{
            "Execução": len(st.session_state.history) + 1,
            "Imagem": img_uri,
            "Classificação": classification,
            "Erro MSE": round(mse, 6),
            "Confiança (%)": confidence_percent,
            "Data/Hora": current_time,
        }])

        st.session_state.history_df = pd.concat(
            [st.session_state.history_df, new_row],
            ignore_index=True
        )

        st.session_state.history.append({
            "classification": classification,
            "mse": mse,
            "confidence": confidence_percent,
            "timestamp": current_time,
        })

    # ==========================================
    # TABS
    # ==========================================
    tab_triagem, tab_pipeline, tab_geracao, tab_historico, tab_feedback, tab_sobre = st.tabs([
        "🔍 Triagem",
        "⚙️ Pipeline & Explicabilidade",
        "🎨 Geração de Imagens",
        "📊 Histórico Operacional",
        "🧑‍⚕️ Feedback Humano & Monitoramento",
        "ℹ️ Sobre o Modelo",
    ])

    # ==========================================
    # TAB 1 — TRIAGEM
    # ==========================================
    with tab_triagem:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagem Original")
            st.image(x[0].squeeze(), clamp=True, width=180)

        with col2:
            st.subheader("Reconstrução VAE")
            st.image(recon[0].squeeze(), clamp=True, width=180)

        st.markdown("---")
        st.subheader("📊 Resultado da Triagem")

        prev_mse = st.session_state.history[-2]["mse"] if len(st.session_state.history) >= 2 else None
        delta_mse = f"{(mse - prev_mse):+.6f}" if prev_mse is not None else None

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Erro de Reconstrução (MSE)", f"{mse:.6f}", delta=delta_mse, delta_color="inverse")
        m2.metric("Classificação", classification)
        m3.metric("Confiança estimada", f"{confidence_percent}%")
        m4.metric("Faixa de confiança", confidence_label)

        st.progress(confidence_percent)

        if color == "green":
            st.success(f"✅ {classification} — {description}")
        elif color == "orange":
            st.warning(f"⚠️ {classification} — {description}")
        else:
            st.error(f"🚨 {classification} — {description}")

        if confidence_type == "success":
            st.success(confidence_message)
        elif confidence_type == "warning":
            st.warning(confidence_message)
        else:
            st.error(confidence_message)

        st.markdown(
            f"""
            <div style="padding:1rem; border-radius:0.5rem;
                        background-color:{color}20; border-left:4px solid {color}; margin-top:0.5rem;">
                <h4 style="color:{color}; margin:0;">Resultado estimado: {classification}</h4>
                <p style="margin:0.5rem 0 0 0;">{description}</p>
                <p style="margin:0.5rem 0 0 0;"><strong>Faixa de confiança:</strong> {confidence_label}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "⚠️ **Importante:** este sistema é apenas um auxiliar de triagem. "
            "Sempre consulte um médico para diagnóstico definitivo."
        )

        st.markdown("---")
        st.subheader("🧑‍⚕️ Validação Humana")

        fc1, fc2 = st.columns(2)

        with fc1:
            if st.button("✅ Classificação correta", use_container_width=True):
                fb_img_uri = image_to_base64_uri(st.session_state.last_result["x"][0])
                st.session_state.feedback_log.append({
                    "Execução": len(st.session_state.feedback_log) + 1,
                    "Imagem": fb_img_uri,
                    "Classificação": classification,
                    "Erro MSE": round(mse, 6),
                    "Confiança (%)": confidence_percent,
                    "Feedback": "Correta",
                    "Data/Hora": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                })
                st.toast("Feedback positivo registrado.")

        with fc2:
            if st.button("❌ Classificação incorreta", use_container_width=True):
                fb_img_uri = image_to_base64_uri(st.session_state.last_result["x"][0])
                st.session_state.feedback_log.append({
                    "Execução": len(st.session_state.feedback_log) + 1,
                    "Imagem": fb_img_uri,
                    "Classificação": classification,
                    "Erro MSE": round(mse, 6),
                    "Confiança (%)": confidence_percent,
                    "Feedback": "Incorreta",
                    "Data/Hora": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                })
                st.toast("Feedback negativo registrado.")

    # ==========================================
    # TAB 2 — PIPELINE
    # ==========================================
    with tab_pipeline:
        st.subheader("⚙️ Como a IA chegou ao resultado")

        pipe1, pipe2 = st.columns(2)

        with pipe1:
            st.markdown("### Etapas do pipeline")
            st.markdown(
                """
                1. **Upload da imagem**
                2. **Conversão para grayscale**
                3. **Redimensionamento para 28x28**
                4. **Reconstrução da imagem pelo VAE**
                5. **Cálculo do erro MSE**
                6. **Classificação com thresholds configuráveis**
                """
            )

        with pipe2:
            st.markdown("### Parâmetros usados nesta execução")
            st.write(f"**Threshold Normal:** {st.session_state.threshold_normal:.3f}")
            st.write(f"**Threshold Borderline:** {st.session_state.threshold_borderline:.3f}")
            st.write(f"**Erro calculado (MSE):** {mse:.6f}")
            st.write(f"**Confiança estimada:** {confidence_percent}%")
            st.write(f"**Data/Hora da análise:** {st.session_state.last_result['timestamp']}")

        st.markdown("---")
        st.markdown("### Interpretação do resultado")
        st.info(
            "O modelo tenta reconstruir a imagem de entrada. Quanto maior a diferença entre "
            "a imagem original e a reconstruída, maior o erro MSE e maior o indício de anomalia."
        )

        st.markdown("### Faixas de confiança adotadas")
        st.markdown(
            """
            - **Alta confiança:** acima de 85%
            - **Média confiança:** entre 60% e 85%
            - **Baixa confiança:** abaixo de 60%
            """
        )

        st.warning(
            "A confiança exibida é uma **estimativa heurística baseada no MSE**. "
            "Ela não representa certeza clínica."
        )

    # ==========================================
    # TAB 3 — GERAÇÃO DE IMAGENS
    # ==========================================
    with tab_geracao:
        st.subheader("🎨 Geração de Imagens Sintéticas")
        st.caption(
            "O decoder do VAE pode gerar novas imagens a partir de amostras do espaço latente aprendido."
        )

        if st.button("🖼️ Gerar imagens sintéticas"):
            generated = generate_new_images(vae, st.session_state.num_generated)
            st.session_state.generated_images = generated
            st.toast(f"{st.session_state.num_generated} imagens geradas com sucesso.")

        if st.session_state.generated_images is not None:
            st.markdown(f"### Galeria ({st.session_state.num_generated} imagens)")
            cols = st.columns(min(st.session_state.num_generated, 4))

            for idx, img in enumerate(st.session_state.generated_images):
                with cols[idx % len(cols)]:
                    st.image(img.squeeze(), clamp=True, caption=f"Imagem {idx + 1}", use_container_width=True)
        else:
            st.info("Clique em **🖼️ Gerar imagens sintéticas** para visualizar novas amostras criadas pelo VAE.")

    # ==========================================
    # TAB 4 — HISTÓRICO OPERACIONAL
    # ==========================================
    with tab_historico:
        st.subheader("📊 Histórico de Análises")
        st.caption(
            "Antes de confiar nos gráficos, inspecione primeiro os dados tabulares e as estatísticas descritivas."
        )

        if not st.session_state.history_df.empty:
            total_exec = len(st.session_state.history_df)
            mean_mse = st.session_state.history_df["Erro MSE"].mean()
            mean_conf = st.session_state.history_df["Confiança (%)"].mean()

            h1, h2, h3 = st.columns(3)
            h1.metric("Total de análises", total_exec)
            h2.metric("Média de MSE", f"{mean_mse:.6f}")
            h3.metric("Média de confiança", f"{mean_conf:.1f}%")

            st.dataframe(
                st.session_state.history_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Imagem": st.column_config.ImageColumn(
                        "Imagem",
                        help="Imagem analisada",
                        width="small",
                    ),
                    "Confiança (%)": st.column_config.ProgressColumn(
                        "Confiança",
                        help="Confiança estimada pelo sistema",
                        min_value=0,
                        max_value=100,
                        format="%d%%",
                    ),
                    "Erro MSE": st.column_config.NumberColumn("Erro MSE", format="%.6f"),
                },
            )

            st.markdown("#### Estatísticas descritivas")
            st.dataframe(
                st.session_state.history_df[["Erro MSE", "Confiança (%)"]].describe().round(6),
                use_container_width=True,
            )

            st.markdown("#### Filtrar histórico por Erro MSE máximo")
            max_mse_filter = st.slider(
                "MSE máximo",
                min_value=0.000, max_value=0.100, value=0.050, step=0.001,
                format="%.3f",
                key="mse_filter",
            )

            if st.button("Aplicar filtro de histórico"):
                st.session_state["filtered_history"] = st.session_state.history_df[
                    st.session_state.history_df["Erro MSE"] <= max_mse_filter
                ]

            if "filtered_history" in st.session_state:
                st.dataframe(
                    st.session_state["filtered_history"],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Imagem": st.column_config.ImageColumn(
                            "Imagem",
                            help="Imagem analisada",
                            width="small",
                        ),
                        "Confiança (%)": st.column_config.ProgressColumn(
                            "Confiança",
                            min_value=0,
                            max_value=100,
                            format="%d%%"
                        ),
                        "Erro MSE": st.column_config.NumberColumn("Erro MSE", format="%.6f"),
                    },
                )

            st.markdown("#### Distribuição das classificações")
            class_dist_df = build_class_distribution_df(st.session_state.history)

            if not class_dist_df.empty:
                fig_class = px.bar(
                    class_dist_df,
                    x="Classificação",
                    y="Quantidade",
                    color="Classificação",
                    title="Distribuição das classificações no histórico",
                )
                st.plotly_chart(fig_class, use_container_width=True)

            st.markdown("#### Evolução do Erro MSE ao longo das execuções")
            fig_mse_evol = px.line(
                st.session_state.history_df,
                x="Execução",
                y="Erro MSE",
                markers=True,
                title="Erro MSE por execução",
            )
            st.plotly_chart(fig_mse_evol, use_container_width=True)

            st.markdown("#### Evolução da Confiança ao longo das execuções")
            fig_conf_evol = px.line(
                st.session_state.history_df,
                x="Execução",
                y="Confiança (%)",
                markers=True,
                title="Confiança estimada por execução",
            )
            st.plotly_chart(fig_conf_evol, use_container_width=True, key="chart_conf_historico")
        else:
            st.info("Ainda não há análises registradas.")

    # ==========================================
    # TAB 5 — FEEDBACK HUMANO & MONITORAMENTO
    # ==========================================
    with tab_feedback:
        st.subheader("🧑‍⚕️ Feedback Humano & Monitoramento")

        feedback_df = build_feedback_df(st.session_state.feedback_log)

        total_feedback = len(feedback_df)

        if total_feedback > 0:
            correct_count = (feedback_df["Feedback"] == "Correta").sum()
            incorrect_count = (feedback_df["Feedback"] == "Incorreta").sum()
            approval_rate = (correct_count / total_feedback) * 100

            fb1, fb2, fb3 = st.columns(3)
            fb1.metric("Feedbacks recebidos", total_feedback)
            fb2.metric("Confirmações positivas", int(correct_count))
            fb3.metric("Taxa de concordância", f"{approval_rate:.1f}%")

            if approval_rate < 70:
                st.warning("⚠️ Possível degradação percebida do modelo: muitos feedbacks negativos foram registrados.")
            else:
                st.success("✅ Monitoramento saudável até o momento, com boa taxa de concordância humana.")

            st.markdown("#### Histórico de feedback")
            st.dataframe(
                feedback_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Imagem": st.column_config.ImageColumn(
                        "Imagem",
                        help="Imagem analisada",
                        width="small",
                    ),
                    "Confiança (%)": st.column_config.ProgressColumn(
                        "Confiança",
                        min_value=0,
                        max_value=100,
                        format="%d%%"
                    ),
                    "Erro MSE": st.column_config.NumberColumn("Erro MSE", format="%.6f"),
                },
            )

            st.markdown("#### Evolução dos feedbacks")
            feedback_plot_df = feedback_df.copy()
            feedback_plot_df["Valor"] = feedback_plot_df["Feedback"].map({
                "Correta": 1,
                "Incorreta": 0
            })

            fig_feedback = px.line(
                feedback_plot_df,
                x="Execução",
                y="Valor",
                markers=True,
                title="Evolução do feedback humano (1 = correta, 0 = incorreta)",
            )
            fig_feedback.update_yaxes(range=[0, 1], tickvals=[0, 1])
            fig_feedback.update_xaxes(dtick=1)
            st.plotly_chart(fig_feedback, use_container_width=True)

            st.markdown("#### Evolução da confiança ao longo das análises")
            confidence_history_df = st.session_state.history_df.copy()
            fig_conf = px.line(
                confidence_history_df,
                x="Execução",
                y="Confiança (%)",
                markers=True,
                title="Confiança estimada por execução",
            )
            st.plotly_chart(fig_conf, use_container_width=True, key="chart_conf_feedback")
        else:
            st.info("Ainda não há feedback suficiente para monitoramento humano.")

    # ==========================================
    # TAB 6 — SOBRE O MODELO
    # ==========================================
    with tab_sobre:
        st.header("ℹ️ Sobre o Modelo VAE")

        st.markdown("""
        ### Arquitetura do Modelo

        **Encoder:**  
        Conv2D(32) → Conv2D(64) → Flatten → Dense(128) → Espaço Latente (z_mean, z_log_var, z)

        **Decoder:**  
        Dense(7×7×64) → Reshape → Conv2DTranspose(64) → Conv2DTranspose(32) → Output(sigmoid)

        ### Como Funciona a Triagem

        1. **Imagens normais:** tendem a produzir menor erro de reconstrução.
        2. **Imagens anômalas:** tendem a produzir maior erro de reconstrução.
        3. **Classificação por thresholds configuráveis:**
           - MSE < Threshold Normal → **NORMAL**
           - Threshold Normal ≤ MSE < Threshold Borderline → **BORDERLINE**
           - MSE ≥ Threshold Borderline → **POSSÍVEL PNEUMONIA**

        ### Limitações

        - Treinado apenas em PneumoniaMNIST (28×28 grayscale)
        - Não substitui diagnóstico médico profissional
        - Sensibilidade depende da qualidade da imagem enviada
        - A confiança exibida é uma estimativa heurística
        """)

        st.markdown("---")
        st.subheader("Estatísticas do Modelo")

        s1, s2 = st.columns(2)
        with s1:
            st.metric("Parâmetros Encoder", f"{vae.encoder.count_params():,}")
            st.metric("Parâmetros Decoder", f"{vae.decoder.count_params():,}")
        with s2:
            st.metric("Total de Parâmetros", f"{vae.count_params():,}")
            st.metric("Dimensão Latente", vae.encoder.output_shape[0][-1])

else:
    st.info("Após enviar a imagem, clique em **🔍 Executar Triagem** para visualizar os resultados.")


# ==============================================
# FOOTER
# ==============================================
st.markdown("---")
st.caption(
    "🔬 **Modelo VAE para Triagem de Pneumonia** | "
    "Desenvolvido com TensorFlow e Streamlit | "
    "Sempre consulte um médico para diagnóstico definitivo."
)
