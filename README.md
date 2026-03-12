# 🔬 VAE PneumoniaMNIST — Triagem de Pneumonia e Geração de Imagens

Este projeto é um **Checkpoint para a disciplina de GenAI & FrontEnd (FIAP)**. Desenvolvemos uma interface interativa utilizando **Streamlit** para a triagem automática de pneumonia, fundamentada em um modelo **VAE (Variational Autoencoder)** treinado sobre o dataset *PneumoniaMNIST*.

O foco principal foi a evolução da experiência do usuário (UX) em sistemas de IA, aplicando conceitos de transparência, gestão de incerteza e feedback humano.

---

## 👥 Grupo (RM)
* **Augusto Oliveira Codo de Sousa** – RM562080
* **Felipe de Oliveira Cabral** – RM561720
* **Gabriel Tonelli Avelino Dos Santos** – RM564705
* **Vinícius Adrian Siqueira de Oliveira** – RM564962
* **Sofia Bueris Netto de Souza** – RM565818

---

## 📚 Sobre o Projeto
A aplicação permite processar exames de raio-X para detectar anomalias através do erro de reconstrução do VAE. Quanto maior a diferença entre a imagem original e a reconstruída pelo modelo, maior a probabilidade de presença de padrões indicativos de pneumonia.

### Diferenciais de UX/AI implementados:
* **Design para Latência:** Uso de indicadores de carregamento e simulação de processos.
* **Gestão de Confiança:** Interface visual que explicita o nível de incerteza do modelo.
* **Human-in-the-loop:** Sistema de feedback para que especialistas validem os resultados.
* **Pipeline Explicitado:** Transparência sobre como os dados percorrem o modelo.

---

## 🚀 Como Executar o Projeto

### 1. Pré-Requisitos
* Python 3.9 ou superior.
* Gerenciador de pacotes `pip`.
* Ambiente virtual (recomendado).

### 2. Instalação
```bash
# Clone o repositório
git clone <SEU_REPOSITORIO_GITHUB>
cd <PASTA_DO_PROJETO>

# Crie e ative o ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Instale as dependências
pip install --upgrade pip
pip install -r requirements.txt
pip install pandas plotly altair Pillow

```

### 3. Treinamento do Modelo

Caso os arquivos em `/models/` não existam, execute o script de treino:

```bash
python train_vae.py

```

*O script baixará o dataset, treinará por 20 épocas e salvará os pesos (`vae_pneumonia.weights.h5`) e configurações.*

### 4. Inicialização do App

```bash
streamlit run app.py

```

Acesse em: `http://localhost:8501`

---

## 🖥️ Funcionalidades e Uso

### **Painel Lateral (Configurações)**

* **Thresholds de Triagem:** Ajuste manual do MSE (Mean Squared Error) para definir o que é Normal ou Borderline.
* **Geração de Imagens:** Define a quantidade de amostras sintéticas a serem geradas.
* **Controle de Sistema:** Simulação de latência e botões para limpeza de cache/reset de sessão.

### **Fluxo de Operação**

1. **Upload:** Envie um raio-X (JPG/PNG).
2. **Execução:** Clique em "Executar Triagem".
3. **Análise:** Visualize a comparação entre a imagem original e a reconstrução do VAE.
4. **Feedback:** Classifique o resultado como correto ou incorreto para alimentar o dashboard de monitoramento.

---

## 🧑‍⚕️ Entenda as Faixas de Confiança

A interface utiliza cores semânticas para indicar a confiabilidade da predição:

| Nível de Confiança | Porcentagem | Recomendação |
| --- | --- | --- |
| **Alta** | > 85% | Resultado consistente com os padrões aprendidos. |
| **Média** | 60% – 85% | Atenção redobrada; indicação de revisão clínica. |
| **Baixa** | < 60% | Incerteza elevada; revisão humana obrigatória. |

---

## 🧑‍🔬 Detalhes Técnicos (Pipeline)

O VAE processa imagens de **28x28 pixels (1 canal)**. O diagnóstico é baseado na métrica de anomalia:

* **Reconstrução Fiel:** Baixo erro (MSE) → Imagem dentro do padrão "Normal".
* **Reconstrução Falha:** Alto erro (MSE) → Presença de padrões desconhecidos (Possível Pneumonia).

---

## 📊 Checklist de Critérios (Rubrica)

* [x] **Interface:** Organizada em Tabs e Sidebar.
* [x] **Interatividade:** Sliders e switches dinâmicos.
* [x] **Latência:** `st.status` e `st.spinner` aplicados.
* [x] **Feedback:** Sistema de histórico operacional e humano.
* [x] **Performance:** Uso de `@st.cache_data` e `@st.cache_resource`.

---

## ⚠️ Importante

Este sistema possui fins **estritamente educacionais**. O modelo não substitui o diagnóstico médico profissional. Sempre consulte um especialista.

## 📄 Licença

Este projeto está sob a licença MIT.

