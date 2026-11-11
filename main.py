"""
main.py — Interface de linha de comando do ShopBot (vendedor virtual).

Chat multiturno no terminal que integra:
  - o agente (agent.loja_agent): create_agent + MemorySaver, sessão fixa por thread_id;
  - a trajetória das ferramentas (agent.loja_eval.TrajetoriaCallbackHandler);
  - a auditoria factual por turno (agent.loja_eval.JuizFactual).

Execução (a partir da raiz do projeto):
    python main.py

Comandos rápidos: sair | ajuda | carrinho | pedidos | limpar_logs
"""

import json
import os
import sys
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from agent.loja_agent import LLM_MODEL, conversar, get_agente  # noqa: E402
from agent.loja_eval import (  # noqa: E402
    JUIZ_MODEL,
    LOGS_DIR,
    JuizFactual,
    TrajetoriaCallbackHandler,
)
from agent.loja_rag import RAG_TOOLS  # noqa: E402
from agent.loja_tools import TOOLS  # noqa: E402

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
THREAD_ID = "sessao_cliente_01"  # sessão fixa: o MemorySaver guarda o histórico por thread_id

# Se True, o juiz recebe as evidências da sessão inteira (e não só do turno). Útil quando as
# respostas se apoiam em consultas de turnos anteriores; o padrão segue a regra "por turno".
JUIZ_INCLUI_HISTORICO = False

COMANDOS_SAIR = {"sair", "exit"}
COMANDOS_AJUDA = {"ajuda", "help"}

TOOLS_POR_NOME = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# Utilidades de terminal
# ---------------------------------------------------------------------------
def configurar_terminal() -> None:
    """Força UTF-8 na saída (acentos) e habilita cores ANSI no Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if os.name == "nt":
        os.system("")  # truque conhecido para ativar sequências ANSI no console do Windows 10+


def discreto(texto: str) -> str:
    """Texto em cinza/esmaecido (só quando a saída é um terminal)."""
    return f"\033[2m{texto}\033[0m" if sys.stdout.isatty() else texto


def formatar(resultado: Any) -> str:
    """Resultados de tools podem ser dict/list (JSON) ou texto."""
    if isinstance(resultado, (dict, list)):
        return json.dumps(resultado, ensure_ascii=False, indent=2)
    return str(resultado)


# ---------------------------------------------------------------------------
# Hook de memória de longo prazo
# ---------------------------------------------------------------------------
def inicializar_memoria_longo_prazo() -> dict:
    """
    Inicializa e carrega o perfil persistente do cliente (data/perfil_cliente.json).
    Exibe o status da memória de longo prazo no console durante a partida do sistema.
    """
    try:
        perfil = carregar_perfil()
        resumo = obter_resumo_perfil(perfil)

        print("[MEMÓRIA DE LONGO PRAZO]: Perfil do cliente carregado.")

        if perfil.nome or perfil.categorias_preferidas or perfil.orcamento_maximo_usd:
            nome_str = perfil.nome if perfil.nome else "Não identificado"

            cats_str = (
                ", ".join(perfil.categorias_preferidas)
                if perfil.categorias_preferidas
                else "Nenhuma"
            )

            orc_str = (
                f"USD ${perfil.orcamento_maximo_usd:.2f}"
                if perfil.orcamento_maximo_usd
                else "Não definido"
            )

            print(f" ├─ Cliente: {nome_str}")
            print(f" ├─ Categorias Preferidas: {cats_str}")
            print(f" └─ Teto Orçamentário: {orc_str}")

        else:
            print(" └─ Perfil inicial limpo (nenhuma preferência registrada ainda).")

        return {
            "status": "sucesso",
            "perfil": perfil,
            "resumo_texto": resumo
        }

    except Exception as e:
        print(f"⚠️ [MEMÓRIA DE LONGO PRAZO]: Aviso ao carregar perfil: {e}")

        return {
            "status": "erro",
            "perfil": None,
            "resumo_texto": "Erro ao carregar perfil de longo prazo."
        }


# ---------------------------------------------------------------------------
# Cabeçalho, status e ajuda
# ---------------------------------------------------------------------------
def exibir_cabecalho() -> None:
    print("=" * 70)
    print("  ShopBot - Vendedor Virtual")
    print("  Projeto final: Agentes Cognitivos com LangChain")
    print("=" * 70)
    print("Comandos rápidos:")
    print("  sair        encerra o chat (salva os logs)")
    print("  ajuda       mostra exemplos de perguntas para testar")
    print("  carrinho    mostra o carrinho (executa a ferramenta ver_carrinho)")
    print("  pedidos     lista os pedidos já faturados")
    print("  limpar_logs esvazia a pasta logs/ para novos testes")
    print("-" * 70)


def verificar_sistemas(memoria: Optional[Any]) -> tuple[list[tuple[str, str]], bool]:
    """Devolve (linhas de status, chave_ok). Marcas: OK, !! (problema), -- (não configurado)."""
    chave_ok = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    status = [
        ("OK" if chave_ok else "!!", "Chave da API Google" + ("" if chave_ok else " NÃO encontrada no .env")),
        ("OK", f"Agente: {LLM_MODEL} | {len(TOOLS) + len(RAG_TOOLS)} ferramentas"),
        ("OK", f"Juiz factual: {JUIZ_MODEL}"),
    ]

    # Monta o índice do RAG já na partida: se o modelo de embeddings estiver errado, aparece aqui
    # e não só na primeira pergunta sobre políticas.
    if chave_ok:
        try:
            from agent.loja_rag import ARQUIVO_POLITICA, get_vectorstore

            if not ARQUIVO_POLITICA.exists():
                raise FileNotFoundError(f"{ARQUIVO_POLITICA} não encontrado")
            total = get_vectorstore().index.ntotal
            status.append(("OK", f"RAG de políticas: {total} trechos indexados"))
        except Exception as e:
            status.append(("!!", f"RAG de políticas indisponível ({type(e).__name__}: {e})"))
    else:
        status.append(("--", "RAG de políticas: não verificado (sem chave)"))

    status.append(("OK", f"Logs em: {LOGS_DIR}"))
    if memoria is not None:
        status.append(("OK", "Memória de longo prazo: ativa"))
    else:
        status.append(("--", "Memória de longo prazo: não configurada (loja_memory.py pendente)"))
    return status, chave_ok


def exibir_status(status: list[tuple[str, str]]) -> None:
    print("Status dos sistemas:")
    for marca, texto in status:
        print(f"  [{marca}] {texto}")
    print("-" * 70)


def exibir_ajuda() -> None:
    print(
        "\nExemplos para testar:\n"
        "  - Quais jaquetas masculinas vocês têm?                    (catálogo)\n"
        "  - Quanto custa o produto 3 em reais?                      (detalhes + câmbio)\n"
        "  - Adiciona 2 unidades do produto 1 no carrinho            (carrinho)\n"
        "  - Qual o prazo do Sedex? Tem frete grátis?                (políticas / RAG)\n"
        "  - Vocês vendem tênis? Consertam celular?                  (fora de escopo)\n"
        "  - Fecha a compra em 3x com o cupom BEMVINDO10             (descontos e parcelas)\n"
        "  - Quero esvaziar o carrinho                               (pede confirmação)\n"
        "  - Quais são os meus pedidos?                              (histórico)\n"
    )


# ---------------------------------------------------------------------------
# Atalhos que executam ferramentas diretamente (sem passar pelo LLM)
# ---------------------------------------------------------------------------
def executar_atalho(nome_tool: str, tentativas_args: list[dict[str, Any]]) -> None:
    """Executa uma ferramenta direto e imprime o resultado.

    Aceita várias formas de argumentos (tentadas em ordem) porque algumas ferramentas têm
    parâmetros opcionais. Não passa pelo agente: não entra na memória nem na trajetória.
    """
    tool = TOOLS_POR_NOME.get(nome_tool)
    if tool is None:
        print(f"[!] Ferramenta '{nome_tool}' não encontrada em agent.loja_tools.TOOLS.")
        return

    ultimo_erro: Optional[Exception] = None
    for args in tentativas_args:
        try:
            print(f"\n[{nome_tool}]\n{formatar(tool.invoke(args))}")
            return
        except Exception as e:  # argumento inválido, falha de rede etc.
            ultimo_erro = e
    print(f"[!] Falha ao executar '{nome_tool}': {type(ultimo_erro).__name__}: {ultimo_erro}")


def limpar_logs(handler: TrajetoriaCallbackHandler) -> None:
    """Esvazia os arquivos de logs/ (JSON vira lista vazia) e zera o histórico em memória."""
    resposta = input("Isso apaga o conteúdo de todos os arquivos em logs/. Confirmar? (s/n): ")
    if resposta.strip().lower() not in {"s", "sim", "y", "yes"}:
        print("Operação cancelada.")
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    arquivos = [p for p in LOGS_DIR.iterdir() if p.is_file()]
    for arquivo in arquivos:
        arquivo.write_text("[]\n" if arquivo.suffix == ".json" else "", encoding="utf-8")
    # Sem isto, o próximo salvar_trajetoria() regravaria o histórico antigo.
    handler.historico.clear()
    handler.passos_turno.clear()
    print(f"Logs esvaziados ({len(arquivos)} arquivo(s)).")


# ---------------------------------------------------------------------------
# Turno de conversa
# ---------------------------------------------------------------------------
def persistir_trajetoria(handler: TrajetoriaCallbackHandler) -> None:
    try:
        handler.salvar_trajetoria()
    except OSError as e:
        print(f"[!] Não foi possível salvar a trajetória: {e}")


def processar_turno(
    agente: Any,
    juiz: Optional[JuizFactual],
    handler: TrajetoriaCallbackHandler,
    mensagem: str,
) -> None:
    """Um turno: agente -> resposta -> juiz -> telemetria -> persistência -> limpeza."""
    try:
        resposta = conversar(agente, mensagem, THREAD_ID, callbacks=[handler])
        print(f"\nShopBot: {resposta}\n")

        ferramentas = [passo["ferramenta"] for passo in handler.passos_turno]
        if juiz is None:
            print(discreto(f"[AUDITORIA JUIZ]: indisponível | Tools usadas: {ferramentas}"))
        else:
            try:
                evidencias = handler.obter_evidencias_turno(
                    incluir_turnos_anteriores=JUIZ_INCLUI_HISTORICO
                )
                v = juiz.avaliar_turno(mensagem, resposta, evidencias)
                alucinacao = "Sim" if v.alucinacao_numerica else "Não"
                print(
                    discreto(
                        f"[AUDITORIA JUIZ]: Veredito = {v.veredito} | "
                        f"Alucinação Numérica = {alucinacao} | Tools usadas: {ferramentas}"
                    )
                )
                if v.veredito == "CONTRADITA" or v.alucinacao_numerica:
                    print(discreto(f"  Justificativa: {v.justificativa}"))
            except Exception as e:  # falha do juiz não derruba o chat
                print(
                    discreto(
                        f"[AUDITORIA JUIZ]: falhou ({type(e).__name__}: {e}) | "
                        f"Tools usadas: {ferramentas}"
                    )
                )

        # HOOK loja_memory: atualizar o perfil/memória de longo prazo do cliente com este turno.

    except KeyboardInterrupt:
        print("\n[Turno interrompido. Pode continuar a conversa ou digitar 'sair'.]")
    except Exception as e:  # rede, cota da API, erro de ferramenta etc.
        print(f"\n[Erro ao processar a mensagem] {type(e).__name__}: {e}")
        print("Tente novamente em instantes. Se persistir, confira a chave/cota da API e a conexão.")
    finally:
        persistir_trajetoria(handler)  # grava mesmo se o turno falhou (passos parciais)
        handler.limpar_turno()


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------
def main() -> int:
    configurar_terminal()
    memoria = inicializar_memoria_longo_prazo()

    exibir_cabecalho()
    print("Verificando sistemas...")
    status, chave_ok = verificar_sistemas(memoria)
    exibir_status(status)

    if not chave_ok:
        print("Defina GOOGLE_API_KEY no arquivo .env (raiz do projeto) e rode novamente.")
        return 1

    try:
        agente = get_agente()
    except Exception as e:
        print(f"[Erro] Não foi possível criar o agente: {type(e).__name__}: {e}")
        return 1

    handler = TrajetoriaCallbackHandler()
    try:
        juiz: Optional[JuizFactual] = JuizFactual()
    except Exception as e:  # o chat segue sem auditoria
        juiz = None
        print(f"[!] Juiz factual indisponível ({type(e).__name__}: {e}). Seguindo sem auditoria.")

    print("Pronto! Converse com o ShopBot (digite 'ajuda' para ver exemplos).")

    while True:
        try:
            entrada = input("\nVocê: ").strip()
        except (KeyboardInterrupt, EOFError):  # Ctrl+C / Ctrl+D no prompt = sair
            print()
            break

        if not entrada:
            continue
        comando = entrada.lower()

        try:
            if comando in COMANDOS_SAIR:
                break
            elif comando in COMANDOS_AJUDA:
                exibir_ajuda()
            elif comando == "carrinho":
                executar_atalho("ver_carrinho", [{}])
            elif comando == "pedidos":
                executar_atalho(
                    "consultar_pedidos", [{}, {"pedido_id": None}, {"pedido_id": ""}]
                )
            elif comando == "limpar_logs":
                limpar_logs(handler)
            else:
                processar_turno(agente, juiz, handler, entrada)
        except KeyboardInterrupt:
            print("\n[Operação interrompida. Digite 'sair' para encerrar.]")
        except Exception as e:  # rede de segurança: o terminal nunca deve cair
            print(f"\n[Erro inesperado] {type(e).__name__}: {e}")

    persistir_trajetoria(handler)
    print(f"\nAté logo! Logs salvos em: {LOGS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())