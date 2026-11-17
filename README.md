# ShopBot — Assistente Cognitivo de E-commerce

O ShopBot é um assistente de e-commerce em CLI que consulta produtos, auxilia no carrinho e responde dúvidas sobre as políticas da loja. O projeto usa LangChain/LangGraph, Google Gemini, ferramentas externas, RAG com FAISS e auditoria das respostas.

Este é um projeto acadêmico apresentado na disciplina de Agentes Cognitivos com Langchain, do MBA em Engenharia de IA, Machine Learning e Deep Learning da INFNET.

## Pré-requisitos e configuração

Use Python com `pip` e uma chave da API Google Gemini. Na raiz do projeto, instale as dependências:

```bash
pip install -r requirements.txt
```

Crie o arquivo `.env` — você pode copiar `.env.example` — e configure:

```dotenv
GOOGLE_API_KEY=sua_chave_aqui
RAG_LIMIAR_DISTANCIA=0.882
```

O modelo do agente é configurável pela variável `LLM_MODEL`; sem ela, o código usa `gemini-3.5-flash-lite`.

## Executar o bot

Inicie a aplicação a partir da raiz do repositório:

```bash
python main.py
```

No chat, use os comandos:

| Comando | Descrição |
|---|---|
| `ajuda` | Exibe exemplos de perguntas e testes. |
| `carrinho` | Consulta o carrinho salvo em `data/carrinho.json`. |
| `pedidos` | Lista os pedidos faturados em `data/pedidos.json`. |
| `limpar_logs` | Solicita confirmação e limpa os arquivos locais em `logs/`. |
| `sair` / `exit` | Encerra a sessão e salva os logs. |

## Módulos

- `main.py`: interface de chat no terminal.
- `agent/loja_agent.py`: agente ReAct com `create_agent` e memória de sessão via `MemorySaver` e `thread_id`.
- `agent/loja_tools.py`: 11 ferramentas para catálogo (FakeStoreAPI), conversão cambial e gestão local de carrinho e pedidos.
- `agent/loja_rag.py`: consulta políticas em `knowledge/politica_loja.md` usando FAISS e limiar L2 configurável.
- `agent/loja_eval.py`: registra trajetórias em `logs/trajetoria_tools.json` e avaliações do Juiz Factual em `logs/avaliacao_juiz.log`.
- `agent/loja_memory.py`: implementa o perfil persistente do cliente, armazenado em `data/perfil_cliente.json`.

## Estrutura do repositório

```text
.
├── agent/
│   ├── loja_agent.py
│   ├── loja_eval.py
│   ├── loja_memory.py
│   ├── loja_rag.py
│   └── loja_tools.py
├── data/                 # carrinho, pedidos e perfil local
├── evidencias/
├── knowledge/
│   └── politica_loja.md
├── logs/                 # auditoria local
├── main.py
├── requirements.txt
└── .env.example
```
