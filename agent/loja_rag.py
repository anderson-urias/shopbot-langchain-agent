"""
loja_rag.py — RAG sobre a Política Comercial da loja (FAISS em memória).

Responsabilidades:
  1. Ler knowledge/politica_loja.md e quebrar em trechos (RecursiveCharacterTextSplitter
     com separadores semânticos de Markdown).
  2. Indexar os trechos num FAISS em memória usando embeddings do Gemini.
  3. Expor a tool `consultar_politicas_loja`, com guardrail anti-alucinação baseado
     na distância L2 do FAISS: se o melhor trecho estiver "longe demais" da pergunta
     (distância > LIMIAR_DISTANCIA_L2), a tool devolve status "sem_cobertura".
  4. Oferecer um modo de calibração do limiar de distância:
         python -m agent.loja_rag --calibrar
     e um modo de teste manual:
         python -m agent.loja_rag --consulta "quanto tempo demora o sedex?"
"""

import argparse
import json
import os
import re
import threading
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ARQUIVO_POLITICA = BASE_DIR / "knowledge" / "politica_loja.md"

# text-embedding-004 foi desligado pelo Google em jan/2026; o substituto
# recomendado é o gemini-embedding-2. Configurável via .env caso precise trocar.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 60
SEPARADORES_MARKDOWN = ["\n#### ", "\n##### ", "\n## ", "\n\n", "\n", " "]

# Quantos trechos recuperar por consulta.
TOP_K = 3

SECAO_PADRAO = "Políticas Gerais"

# Limiar de distância L2 (menor = mais próximo). Valor PROVISÓRIO: rode
# `python -m agent.loja_rag --calibrar` e copie o valor sugerido para
# RAG_LIMIAR_DISTANCIA no .env. Com normalize_L2=True a distância é o L2 ao
# quadrado de vetores unitários (= 2 - 2*cos): 0 = idêntico, maior = mais distante.
LIMIAR_DISTANCIA_L2 = float(os.getenv("RAG_LIMIAR_DISTANCIA", "0.70"))


# ---------------------------------------------------------------------------
# Indexação
# ---------------------------------------------------------------------------
def _extrair_titulos(texto: str) -> list[str]:
    """Devolve os títulos Markdown (linhas que começam com '#') de um trecho, sem os '#'."""
    return [
        linha.strip().lstrip("#").strip()
        for linha in texto.splitlines()
        if linha.strip().startswith("#")
    ]


def _carregar_documentos() -> list[Document]:
    """Lê o markdown, quebra em chunks e injeta `metadata["secao"]` em cada um."""
    if not ARQUIVO_POLITICA.exists():
        raise FileNotFoundError(f"Base de conhecimento não encontrada: {ARQUIVO_POLITICA}")

    texto = ARQUIVO_POLITICA.read_text(encoding="utf-8")
    # Remove as linhas "---" (separadores visuais que não carregam significado).
    texto = re.sub(r"^\s*---\s*$", "", texto, flags=re.MULTILINE)
    # Descarta o preâmbulo (título + aviso ao assistente): é meta-texto e poderia
    # casar com perguntas aleatórias, enfraquecendo o guardrail de cobertura.
    inicio = texto.find("\n## ")
    if inicio != -1:
        texto = texto[inicio:]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARADORES_MARKDOWN,
    )
    chunks = splitter.create_documents([texto], metadatas=[{"arquivo": ARQUIVO_POLITICA.name}])

    # Enriquecimento de metadados: título da seção de cada chunk.
    documentos: list[Document] = []
    secao_corrente = SECAO_PADRAO
    for chunk in chunks:
        titulos = _extrair_titulos(chunk.page_content)
        # Chunk com título próprio usa o primeiro; chunk de continuação (sem título,
        # porque a seção foi quebrada ao meio) herda a seção do chunk anterior.
        chunk.metadata["secao"] = titulos[0] if titulos else secao_corrente
        if titulos:
            secao_corrente = titulos[-1]

        # Ignora chunks "órfãos" que só contêm cabeçalho (ex.: "## 2. ..." seguido de "### 2.1").
        tem_corpo = any(
            linha.strip() and not linha.strip().startswith("#")
            for linha in chunk.page_content.splitlines()
        )
        if tem_corpo:
            documentos.append(chunk)

    if not documentos:
        raise ValueError("Nenhum trecho indexável encontrado em politica_loja.md")
    return documentos


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    # A chave é lida de GOOGLE_API_KEY (carregada do .env).
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


_VECTORSTORE: Optional[FAISS] = None
_LOCK = threading.Lock()


def get_vectorstore() -> FAISS:
    """Constrói o índice na primeira chamada e reutiliza nas seguintes (singleton thread-safe)."""
    global _VECTORSTORE
    if _VECTORSTORE is None:
        with _LOCK:
            if _VECTORSTORE is None:  # checagem dupla: outra thread pode ter construído enquanto esperávamos
                _VECTORSTORE = FAISS.from_documents(
                    _carregar_documentos(),
                    _get_embeddings(),
                    normalize_L2=True,  # vetores unitários -> distâncias comparáveis entre consultas
                )
    return _VECTORSTORE


def buscar_trechos(pergunta: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    """Devolve [(Document, distância L2)] ordenado do mais próximo ao mais distante."""
    return get_vectorstore().similarity_search_with_score(pergunta, k=k)


# ---------------------------------------------------------------------------
# Tool do agente
# ---------------------------------------------------------------------------
class ConsultarPoliticasInput(BaseModel):
    pergunta: str = Field(
        ...,
        min_length=3,
        description=(
            "Pergunta do cliente sobre frete, prazos, trocas, devoluções, garantia, "
            "formas de pagamento, parcelamento, cupons ou sobre quais produtos/serviços "
            "a loja vende ou não vende. Use a pergunta completa, em português."
        ),
    )


@tool("consultar_politicas_loja", args_schema=ConsultarPoliticasInput)
def consultar_politicas_loja(pergunta: str) -> dict:
    """Consulta a Política Comercial oficial da loja (frete, trocas e devoluções,
    garantia, pagamento/parcelamento/cupons e limites de escopo do que a loja vende).
    Use SEMPRE que o cliente perguntar sobre regras, prazos ou políticas da loja, ou
    sobre produtos/serviços que a loja pode não oferecer. Responda apenas com o que
    esta ferramenta retornar; se o status for "sem_cobertura", não invente a resposta."""
    try:
        resultados = buscar_trechos(pergunta)
        # float(): o FAISS devolve numpy.float32, que não é serializável em JSON.
        melhor_distancia = float(min(dist for _, dist in resultados))
    except Exception as e:  # falha de API/rede não deve derrubar o agente
        return {
            "status": "erro",
            "mensagem": f"Não foi possível consultar as políticas agora ({type(e).__name__}: {e}).",
        }

    if melhor_distancia > LIMIAR_DISTANCIA_L2:
        return {
            "status": "sem_cobertura",
            "mensagem": "Este tema não consta nas políticas oficiais da loja.",
            "distancia_l2": round(melhor_distancia, 4),
        }

    # Só repassa ao LLM os trechos que também estão dentro do limiar.
    trechos = [(doc, dist) for doc, dist in resultados if dist <= LIMIAR_DISTANCIA_L2]
    return {
        "status": "sucesso",
        "total_trechos": len(trechos),
        "distancia_l2": round(melhor_distancia, 4),
        "trechos": [
            {
                "secao": doc.metadata.get("secao", SECAO_PADRAO),
                "conteudo": doc.page_content.strip(),
            }
            for doc, _ in trechos
        ],
    }


# Lista exportada para o orquestrador (loja_agent.py): TOOLS + RAG_TOOLS.
RAG_TOOLS = [consultar_politicas_loja]


# ---------------------------------------------------------------------------
# Calibração do limiar de distância
# ---------------------------------------------------------------------------
# (pergunta, coberta_pela_politica)
CONSULTAS_CALIBRACAO: list[tuple[str, bool]] = [
    # Cobertas: o documento tem a resposta (inclusive as de "não vendemos / não fazemos")
    ("Quanto tempo demora a entrega pelo Sedex?", True),
    ("A partir de que valor o frete é grátis?", True),
    ("Posso devolver um produto depois que recebi?", True),
    ("Como funciona a garantia dos eletrônicos?", True),
    ("Dá para parcelar no cartão sem juros?", True),
    ("O cupom BEMVINDO10 ainda vale?", True),
    ("Vocês vendem tênis?", True),
    ("Vocês consertam celular na loja?", True),
    ("A garantia cobre se o fone caiu na água?", True),
    # Sem cobertura: assuntos que o documento não trata
    ("Qual é a capital da França?", False),
    ("Me passa uma receita de bolo de cenoura", False),
    ("Quem ganhou a Copa do Mundo de 2022?", False),
    ("Como investir em ações na bolsa?", False),
    ("Como configurar o wifi do meu roteador?", False),
    ("Qual a previsão do tempo para amanhã?", False),
    ("Explique o que é fotossíntese", False),
]


def calibrar() -> float:
    """Mede a distância L2 do melhor trecho para consultas cobertas e não cobertas
    e sugere o limiar que melhor separa os dois grupos."""
    resultados = []  # (pergunta, coberta, distância, local)
    for pergunta, coberta in CONSULTAS_CALIBRACAO:
        doc, dist = buscar_trechos(pergunta, k=1)[0]
        dist = float(dist)
        local = doc.metadata.get("secao", SECAO_PADRAO)
        resultados.append((pergunta, coberta, dist, local))
        marca = "COBERTA    " if coberta else "SEM COBERT."
        print(f"{marca} dist={dist:.3f} | {pergunta}  ->  {local}")

    dist_cobertas = [d for _, c, d, _ in resultados if c]
    dist_nao_cobertas = [d for _, c, d, _ in resultados if not c]
    print(f"\nMaior distância entre as COBERTAS:     {max(dist_cobertas):.3f}")
    print(f"Menor distância entre as SEM COBERTURA: {min(dist_nao_cobertas):.3f}")

    # Testa os pontos médios entre distâncias vizinhas e fica com o de maior acurácia.
    todas = sorted(d for _, _, d, _ in resultados)
    candidatos = [(a + b) / 2 for a, b in zip(todas, todas[1:])]

    def acuracia(limiar: float) -> float:
        acertos = sum((d <= limiar) == coberta for _, coberta, d, _ in resultados)
        return acertos / len(resultados)

    melhor = max(candidatos, key=acuracia)
    print(f"Limiar sugerido: {melhor:.3f} (acurácia no conjunto: {acuracia(melhor):.0%})")
    if max(dist_cobertas) >= min(dist_nao_cobertas):
        print("ATENÇÃO: os grupos se sobrepõem; revise as consultas ou o texto da política.")
    print(f"\nAdicione ao .env:  RAG_LIMIAR_DISTANCIA={melhor:.3f}")
    return melhor


# ---------------------------------------------------------------------------
# CLI de teste
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste e calibração do RAG da loja")
    parser.add_argument("--calibrar", action="store_true", help="calibra o limiar de distância")
    parser.add_argument("--consulta", type=str, help="executa a tool com uma pergunta")
    args = parser.parse_args()

    if args.calibrar:
        calibrar()
    elif args.consulta:
        resposta = consultar_politicas_loja.invoke({"pergunta": args.consulta})
        print(json.dumps(resposta, ensure_ascii=False, indent=2))
    else:
        parser.print_help()