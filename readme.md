
# Chat com URLs e PDFs

Projeto ChatGPT com o conhecimento adquirido de websites por meio de URLs salva em arquivo, e conhecimento adquirido de PDFs por meio de URLs salva em arquivo.

## Instalação
Tenha Python instalado em seu sistema e clone este projeto.

Instale as bibliotecas:

```bash
pip install -r requirements.txt
```

Crie o arquivo .streamlit/secrets.toml dentro da pasta src deste projeto com as variáveis:

```bash
OPENAI_API_KEY="sua_chave_api"
PERSISTENT_VECTORSTORE="False"
FILE_PATH="./"
```

## Uso
Para rodar digite:

```bash
streamlit run app.py
```


