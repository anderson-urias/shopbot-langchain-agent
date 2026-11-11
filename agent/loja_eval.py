"""
loja_eval.py — Auditoria e verificação factual locais do ShopBot (sem LangSmith).

Duas peças:
  1. TrajetoriaCallbackHandler: callback do LangChain que registra cada ferramenta acionada
     (nome, argumentos, saída, horário) e salva a trajetória em logs/trajetoria_tools.json.
  2. JuizFactual: LLM-as-a-Judge que compara a resposta do agente com as saídas das
     ferramentas/RAG (as "evidências") e emite um veredito estruturado
     (SUSTENTADA, CONTRADITA ou NEUTRA), gravado em logs/avaliacao_juiz.log.

Fluxo típico por turno (ex.: no main.py):
    handler = TrajetoriaCallbackHandler()          # uma vez por sessão
    juiz = JuizFactual()                           # uma vez por sessão

    resposta = conversar(agente, pergunta, thread_id, callbacks=[handler])
    verificacao = juiz.avaliar_turno(pergunta, resposta, handler.obter_evidencias_turno())
    handler.salvar_trajetoria()
    handler.limpar_turno()

Teste simulado (não precisa de API): python -m agent.loja_eval
"""

import json
import os
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
os.makedirs(LOGS_DIR, exist_ok=True)  # garante logs/ mesmo se o script rodar de outra pasta

# O juiz usa o mesmo modelo do agente por padrão; configurável via .env.
JUIZ_MODEL = os.getenv("JUIZ_MODEL", os.getenv("LLM_MODEL", "gemini-3.5-flash-lite"))
JUIZ_TEMPERATURE = 0.0

# O Gemini 3.5 Flash-Lite ignora temperature (amostragem fixa) e a lib avisa a cada chamada.
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*", category=UserWarning)

SEM_EVIDENCIAS = "Nenhuma ferramenta acionada neste turno."


def _resolver_caminho(caminho: str | Path) -> Path:
    """Caminhos relativos (ex.: 'logs/x.json') são resolvidos a partir da raiz do projeto."""
    caminho = Path(caminho)
    return caminho if caminho.is_absolute() else BASE_DIR / caminho


def _agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 1. Callback de trajetória
# ---------------------------------------------------------------------------
class TrajetoriaCallbackHandler(BaseCallbackHandler):
    """Registra as ferramentas acionadas pelo agente, turno a turno.

    Cada chamada de ferramenta vira UM passo (dict) com os dados capturados no início
    (nome, argumentos, horário) e no fim (saída, horário). O pareamento é feito pelo
    `run_id`, porque o Gemini pode chamar várias ferramentas em paralelo no mesmo passo.
    """

    def __init__(self) -> None:
        super().__init__()
        self.historico: list[dict[str, Any]] = []  # todos os passos da sessão
        self.passos_turno: list[dict[str, Any]] = []  # passos do turno corrente
        self.turno_atual: int = 1
        self._abertos: dict[str, dict[str, Any]] = {}  # run_id -> passo aguardando saída
        self._lock = threading.Lock()  # tools paralelas chamam o callback em threads distintas

    # -- captura ----------------------------------------------------------
    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        nome = (serialized or {}).get("name") or kwargs.get("name") or "desconhecida"
        # `inputs` traz os argumentos já estruturados (dict); input_str é a versão em texto.
        inputs = kwargs.get("inputs")
        passo: dict[str, Any] = {
            "turno": self.turno_atual,
            "ferramenta": nome,
            "argumentos": inputs if isinstance(inputs, dict) else input_str,
            "inicio": _agora(),
            "saida": None,
            "fim": None,
        }
        with self._lock:
            self._abertos[str(kwargs.get("run_id"))] = passo
            self.passos_turno.append(passo)
            self.historico.append(passo)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        with self._lock:
            passo = self._abertos.pop(str(kwargs.get("run_id")), None)
            if passo is None:  # fim sem início registrado: guarda mesmo assim, para auditoria
                passo = {
                    "turno": self.turno_atual,
                    "ferramenta": kwargs.get("name") or "desconhecida",
                    "argumentos": None,
                    "inicio": None,
                    "saida": None,
                    "fim": None,
                }
                self.passos_turno.append(passo)
                self.historico.append(passo)
            passo["saida"] = self._saida_como_texto(output)
            passo["fim"] = _agora()

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Se a ferramenta falhar, o erro vira a 'saída' (o juiz precisa enxergá-lo)."""
        with self._lock:
            passo = self._abertos.pop(str(kwargs.get("run_id")), None)
            if passo is not None:
                passo["saida"] = f"ERRO: {type(error).__name__}: {error}"
                passo["fim"] = _agora()

    @staticmethod
    def _saida_como_texto(output: Any) -> str:
        """Dentro do agente, `output` costuma ser um ToolMessage; usamos só o conteúdo,
        senão str(output) traria também metadados (tool_call_id, status etc.)."""
        conteudo = getattr(output, "content", output)
        if isinstance(conteudo, str):
            return conteudo
        try:
            return json.dumps(conteudo, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(conteudo)

    # -- consulta e persistência -----------------------------------------
    def obter_evidencias_turno(self, incluir_turnos_anteriores: bool = False) -> str:
        """Texto com as ferramentas acionadas no turno e suas saídas (entrada do juiz).

        Com `incluir_turnos_anteriores=True` inclui a sessão inteira: útil quando a
        resposta se apoia em algo consultado em turnos anteriores (ex.: "o produto que
        você adicionou"), para o juiz não marcar isso como afirmação sem respaldo.
        """
        passos = self.historico if incluir_turnos_anteriores else self.passos_turno
        if not passos:
            return SEM_EVIDENCIAS

        blocos = []
        for i, passo in enumerate(passos, start=1):
            argumentos = json.dumps(passo["argumentos"], ensure_ascii=False, default=str)
            saida = passo["saida"] if passo["saida"] is not None else "(sem saída registrada)"
            blocos.append(
                f"[{i}] Ferramenta: {passo['ferramenta']}\n"
                f"    Argumentos: {argumentos}\n"
                f"    Saída: {saida}"
            )
        return "\n".join(blocos)

    def salvar_trajetoria(self, caminho_arquivo: str = "logs/trajetoria_tools.json") -> Path:
        """Grava o histórico acumulado da sessão em JSON (indent=2, UTF-8, sem escapar acentos)."""
        caminho = _resolver_caminho(caminho_arquivo)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conteudo = json.dumps(self.historico, indent=2, ensure_ascii=False, default=str)
        caminho.write_text(conteudo, encoding="utf-8")
        return caminho

    def limpar_turno(self) -> None:
        """Zera os passos do turno corrente e avança o contador (o histórico é preservado)."""
        with self._lock:
            self.passos_turno = []
            self._abertos.clear()
            self.turno_atual += 1


# ---------------------------------------------------------------------------
# 2. Juiz factual (LLM-as-a-Judge)
# ---------------------------------------------------------------------------
class VerificacaoTurno(BaseModel):
    """Resultado estruturado da verificação factual de um turno."""

    afirmacoes: list[str] = Field(
        description="Principais afirmações factuais feitas na resposta do agente "
        "(preços, descontos, prazos, regras, estoque, status de ações)."
    )
    veredito: Literal["SUSTENTADA", "CONTRADITA", "NEUTRA"] = Field(
        description="SUSTENTADA: todas as afirmações factuais são apoiadas pelas evidências. "
        "CONTRADITA: a resposta contradiz as evidências ou afirma fatos sem nenhum respaldo nelas. "
        "NEUTRA: resposta puramente cordial/informativa, sem alegações factuais que dependam de ferramentas."
    )
    alucinacao_numerica: bool = Field(
        description="True se a resposta cita valores, preços, descontos, prazos ou quantidades "
        "que NÃO aparecem nas saídas das ferramentas."
    )
    justificativa: str = Field(description="Explicação sucinta e objetiva do veredito.")


PROMPT_JUIZ = """\
Você é um auditor factual rigoroso de um assistente de e-commerce (ShopBot). Você recebe:
(1) a pergunta do cliente, (2) a resposta do agente e (3) as EVIDÊNCIAS: as saídas reais das \
ferramentas (catálogo, carrinho, pedidos, câmbio, política da loja via RAG) acionadas no turno.

Regras de julgamento:
- As evidências são a ÚNICA fonte de verdade. Não use conhecimento próprio para validar fatos.
- Liste em `afirmacoes` as principais afirmações factuais da resposta do agente.
- SUSTENTADA: todas as afirmações factuais estão apoiadas pelas evidências. Uma resposta que \
diz que algo "não consta nas políticas" ou "está fora do escopo da loja" é SUSTENTADA quando a \
evidência trouxe status "sem_cobertura" ou um trecho que diz que a loja não vende/faz aquilo.
- CONTRADITA: a resposta contradiz as evidências, OU afirma fatos que dependem de ferramentas \
(preço, prazo, regra, estoque, total, status de pedido) sem nenhum respaldo nas evidências. Se \
houver qualquer afirmação factual não sustentada, NÃO use SUSTENTADA.
- NEUTRA: a resposta é só cordial/procedimental (saudação, pedido de esclarecimento, oferta de \
ajuda) e não faz nenhuma alegação factual que dependa de ferramentas.
- alucinacao_numerica = true se a resposta cita qualquer valor, preço, desconto, prazo, \
quantidade ou parcela que não esteja nas evidências. Ignore números que o próprio cliente \
mencionou na pergunta e diferenças de formatação (ex.: "109,95" e "109.95" são o mesmo valor; \
"US$ 22,30" e "22.3" também).
- Se as evidências forem "Nenhuma ferramenta acionada neste turno.", qualquer fato dependente \
de ferramenta na resposta é não sustentado.
- Seja objetivo. A justificativa deve citar o dado específico que sustenta ou contradiz.
"""


class JuizFactual:
    """Avalia se a resposta do agente é SUSTENTADA, CONTRADITA ou NEUTRA frente às evidências."""

    def __init__(
        self,
        avaliador: Optional[Runnable] = None,
        caminho_log: str = "logs/avaliacao_juiz.log",
    ) -> None:
        # `avaliador` existe só para injetar um juiz falso em testes sem API;
        # em uso normal fica None e o juiz é o Gemini com saída estruturada.
        if avaliador is None:
            llm = ChatGoogleGenerativeAI(model=JUIZ_MODEL, temperature=JUIZ_TEMPERATURE)
            avaliador = llm.with_structured_output(VerificacaoTurno)
        self._avaliador = avaliador
        self._caminho_log = _resolver_caminho(caminho_log)
        self._lock = threading.Lock()

    def avaliar_turno(
        self, pergunta_usuario: str, resposta_agente: str, evidencias_tools: str
    ) -> VerificacaoTurno:
        mensagens = [
            SystemMessage(content=PROMPT_JUIZ),
            HumanMessage(
                content=(
                    f"### PERGUNTA DO CLIENTE\n{pergunta_usuario}\n\n"
                    f"### RESPOSTA DO AGENTE\n{resposta_agente}\n\n"
                    f"### EVIDÊNCIAS DAS FERRAMENTAS\n{evidencias_tools}"
                )
            ),
        ]
        resultado = self._avaliador.invoke(mensagens)

        if isinstance(resultado, dict):
            resultado = VerificacaoTurno.model_validate(resultado)
        if not isinstance(resultado, VerificacaoTurno):
            # Não mascaramos como NEUTRA: um veredito falso estragaria a auditoria.
            raise ValueError("O juiz não retornou uma VerificacaoTurno válida.")

        self._registrar_log(pergunta_usuario, resposta_agente, evidencias_tools, resultado)
        return resultado

    def _registrar_log(
        self, pergunta: str, resposta: str, evidencias: str, v: VerificacaoTurno
    ) -> None:
        afirmacoes = "\n".join(f"  - {a}" for a in v.afirmacoes) or "  (nenhuma)"
        bloco = (
            f"{'=' * 72}\n"
            f"[{_agora()}] AVALIAÇÃO DO JUIZ FACTUAL (modelo: {JUIZ_MODEL})\n"
            f"Pergunta do cliente : {pergunta}\n"
            f"Resposta do agente  : {resposta}\n"
            f"Evidências (tools)  :\n{evidencias}\n"
            f"Afirmações:\n{afirmacoes}\n"
            f"Veredito            : {v.veredito}\n"
            f"Alucinação numérica : {'SIM' if v.alucinacao_numerica else 'NÃO'}\n"
            f"Justificativa       : {v.justificativa}\n"
        )
        self._caminho_log.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._caminho_log.open("a", encoding="utf-8") as f:
            f.write(bloco + "\n")


# ---------------------------------------------------------------------------
# Teste simulado (execução falsa)
# ---------------------------------------------------------------------------
def _avaliador_simulado(mensagens: list) -> VerificacaoTurno:
    """Juiz FALSO para testar sem API: marca alucinação numérica se a resposta tiver
    algum número ausente das evidências. Só serve para demonstrar o fluxo."""
    import re

    texto = mensagens[-1].content
    resposta = re.search(r"### RESPOSTA DO AGENTE\n(.*?)\n\n### EVIDÊNCIAS", texto, re.S).group(1)
    evidencias = texto.split("### EVIDÊNCIAS DAS FERRAMENTAS\n", 1)[1]

    def numeros(s: str) -> set[float]:
        return {float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", s)}

    inventados = numeros(resposta) - numeros(evidencias)
    if inventados:
        return VerificacaoTurno(
            afirmacoes=["(simulado) preço citado na resposta"],
            veredito="CONTRADITA",
            alucinacao_numerica=True,
            justificativa=f"(simulado) valores ausentes das evidências: {sorted(inventados)}",
        )
    return VerificacaoTurno(
        afirmacoes=["(simulado) preço citado na resposta"],
        veredito="SUSTENTADA",
        alucinacao_numerica=False,
        justificativa="(simulado) todos os valores citados constam nas saídas das ferramentas.",
    )


if __name__ == "__main__":
    from uuid import uuid4

    from langchain_core.runnables import RunnableLambda

    usar_api = bool(os.getenv("GOOGLE_API_KEY"))
    print(f"Modo: {'REAL (juiz Gemini)' if usar_api else 'SIMULADO (sem GOOGLE_API_KEY)'}\n")

    handler = TrajetoriaCallbackHandler()
    juiz = JuizFactual() if usar_api else JuizFactual(avaliador=RunnableLambda(_avaliador_simulado))

    pergunta = "Quanto custa a camiseta slim fit (produto 2)?"
    saida_tool = json.dumps({"id": 2, "titulo": "Mens Casual Premium Slim Fit T-Shirts", "preco_usd": 22.3})

    respostas = [
        ("Resposta FIEL", "A Mens Casual Premium Slim Fit T-Shirts custa US$ 22,30."),
        ("Resposta com PREÇO INVENTADO", "A Mens Casual Premium Slim Fit T-Shirts custa US$ 89,90."),
    ]

    for rotulo, resposta in respostas:
        # Simula o que o LangChain faz durante a execução da ferramenta:
        run_id = uuid4()
        handler.on_tool_start(
            {"name": "obter_detalhes_produto"},
            "{'produto_id': 2}",
            run_id=run_id,
            inputs={"produto_id": 2},
        )
        handler.on_tool_end(saida_tool, run_id=run_id)

        evidencias = handler.obter_evidencias_turno()
        print(f"--- {rotulo} ---\nEvidências:\n{evidencias}\n")

        v = juiz.avaliar_turno(pergunta, resposta, evidencias)
        print(f"Veredito: {v.veredito} | alucinação numérica: {v.alucinacao_numerica}")
        print(f"Justificativa: {v.justificativa}\n")
        handler.limpar_turno()

    # Turno sem ferramentas: evidências vazias
    print(f"Sem tools -> {handler.obter_evidencias_turno()!r}")

    caminho = handler.salvar_trajetoria()
    print(f"Trajetória salva em: {caminho}")
    print(f"Log do juiz em:      {juiz._caminho_log}")