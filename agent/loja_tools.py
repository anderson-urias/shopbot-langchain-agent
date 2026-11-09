import json
import os
import time
import uuid
from datetime import datetime

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field, TypeAdapter, computed_field

# ==========================================
# CONFIGURAÇÃO
# ==========================================
API_URL = "https://fakestoreapi.com/products"
CAMBIO_URL = "https://api.exchangerate-api.com/v4/latest/USD"
CAMBIO_TTL_SEGUNDOS = 600  # cotação fica em cache por 10 minutos

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CARRINHO_FILE = os.path.join(DATA_DIR, "carrinho.json")
PEDIDOS_FILE = os.path.join(DATA_DIR, "pedidos.json")
os.makedirs(DATA_DIR, exist_ok=True)

_cache_cambio = {"taxa": None, "ts": 0.0}


# ==========================================
# MODELOS (validação dos dados persistidos)
# ==========================================
class ItemCarrinho(BaseModel):
    id: int
    titulo: str
    preco_unitario_usd: float = Field(ge=0)
    quantidade: int = Field(ge=1)

    @computed_field
    @property
    def subtotal_usd(self) -> float:
        # Sempre calculado em código: o LLM nunca informa preços ou subtotais.
        return round(self.preco_unitario_usd * self.quantidade, 2)


class Pedido(BaseModel):
    pedido_id: str
    data_hora: str
    itens: list[ItemCarrinho]

    @computed_field
    @property
    def total_usd(self) -> float:
        return round(sum(i.subtotal_usd for i in self.itens), 2)


# ==========================================
# SCHEMAS DE ENTRADA DAS TOOLS
# ==========================================
class BuscarProdutosArgs(BaseModel):
    termo_ou_categoria: str = Field(
        default="",
        description=(
            "Categoria exata ('men's clothing', 'women's clothing', 'jewelery', "
            "'electronics') ou termo genérico (ex: 'jacket', 'gold'). "
            "Vazio retorna os primeiros produtos do catálogo."
        ),
    )


class AdicionarCarrinhoArgs(BaseModel):
    produto_id: int = Field(ge=1, description="ID do produto no catálogo.")
    quantidade: int = Field(default=1, ge=1, description="Quantidade a adicionar.")


class AtualizarQuantidadeArgs(BaseModel):
    produto_id: int = Field(ge=1, description="ID do produto que já está no carrinho.")
    nova_quantidade: int = Field(
        ge=0, description="Nova quantidade total do item. Use 0 para remover o item."
    )


class AtualizarPedidoArgs(BaseModel):
    pedido_id: str = Field(description="ID do pedido, no formato PED-AAAAMMDDHHMMSS-xxxx.")
    produto_id: int = Field(ge=1, description="ID do produto dentro do pedido.")
    nova_quantidade: int = Field(
        ge=0, description="Nova quantidade total do item. Use 0 para remover o item do pedido."
    )


class ExcluirPedidoArgs(BaseModel):
    pedido_id: str = Field(description="ID do pedido a excluir.")
    confirmar: bool = Field(
        default=False,
        description="Só use True depois que o usuário confirmar explicitamente a exclusão.",
    )


class ConfirmarArgs(BaseModel):
    confirmar: bool = Field(
        default=False,
        description="Só use True depois que o usuário confirmar explicitamente a ação.",
    )


class ConverterPrecoArgs(BaseModel):
    preco_usd: float = Field(ge=0, description="Valor em dólares, exatamente como veio de outra tool.")


# ==========================================
# FUNÇÕES UTILITÁRIAS (persistência)
# ==========================================
def _quarentena(caminho: str) -> None:
    """Move um arquivo corrompido para .corrompido, em vez de sobrescrevê-lo."""
    try:
        os.replace(caminho, caminho + ".corrompido")
    except OSError:
        pass


def _ler_lista(caminho: str, modelo: type[BaseModel]) -> list:
    """Lê uma lista de modelos do disco. Se o arquivo estiver inválido, isola e devolve []."""
    if not os.path.exists(caminho):
        return []
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)
        return TypeAdapter(list[modelo]).validate_python(dados)
    except ValueError:  # JSONDecodeError e ValidationError herdam de ValueError
        _quarentena(caminho)
        return []


def _salvar_lista(caminho: str, itens: list) -> None:
    """Grava de forma atômica (arquivo temporário + replace)."""
    tmp = caminho + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump([i.model_dump() for i in itens], f, indent=2, ensure_ascii=False)
    os.replace(tmp, caminho)


def _ler_carrinho() -> list[ItemCarrinho]:
    return _ler_lista(CARRINHO_FILE, ItemCarrinho)


def _salvar_carrinho(itens: list[ItemCarrinho]) -> None:
    _salvar_lista(CARRINHO_FILE, itens)


def _ler_pedidos() -> list[Pedido]:
    return _ler_lista(PEDIDOS_FILE, Pedido)


def _salvar_pedidos(pedidos: list[Pedido]) -> None:
    _salvar_lista(PEDIDOS_FILE, pedidos)


def _obter_cotacao_brl() -> float:
    """Retorna a cotação USD→BRL, usando cache para evitar chamadas repetidas."""
    agora = time.time()
    if _cache_cambio["taxa"] and agora - _cache_cambio["ts"] < CAMBIO_TTL_SEGUNDOS:
        return _cache_cambio["taxa"]

    resp = requests.get(CAMBIO_URL, timeout=6)
    resp.raise_for_status()
    taxa = resp.json().get("rates", {}).get("BRL")
    if not taxa:
        raise ValueError("Cotação BRL ausente na resposta da API de câmbio.")

    _cache_cambio.update(taxa=taxa, ts=agora)
    return taxa


# ==========================================
# 1. CATÁLOGO (API EXTERNA)
# ==========================================
@tool(args_schema=BuscarProdutosArgs)
def buscar_produtos_catalogo(termo_ou_categoria: str = "") -> dict:
    """
    Busca produtos no catálogo online (FakeStoreAPI).
    Retorna id, título, preço em dólares, categoria e nota de até 6 produtos.
    """
    try:
        resp = requests.get(API_URL, timeout=8)
        resp.raise_for_status()
        produtos = resp.json()

        termo = termo_ou_categoria.strip().lower()
        if termo:
            produtos = [
                p for p in produtos
                if termo in p["title"].lower()
                or termo in p["category"].lower()
                or termo in p["description"].lower()
            ]
        else:
            produtos = produtos[:5]

        if not produtos:
            return {"status": "vazio", "mensagem": f"Nenhum produto encontrado para '{termo_ou_categoria}'."}

        return {
            "status": "sucesso",
            "produtos": [
                {
                    "id": p["id"],
                    "titulo": p["title"],
                    "preco_usd": round(p["price"], 2),
                    "categoria": p["category"],
                    "nota": p.get("rating", {}).get("rate", "N/A"),
                }
                for p in produtos[:6]
            ],
        }
    except Exception as e:
        return {"status": "erro", "mensagem": f"Falha ao consultar a API: {e}"}


# ==========================================
# 2. CARRINHO
# ==========================================
@tool(args_schema=AdicionarCarrinhoArgs)
def adicionar_ao_carrinho(produto_id: int, quantidade: int = 1) -> dict:
    """
    Adiciona um produto ao carrinho pelo ID do catálogo. O nome e o preço são
    obtidos da API, nunca informados pelo usuário. Se o produto já estiver no
    carrinho, a quantidade é somada.
    """
    try:
        resp = requests.get(f"{API_URL}/{produto_id}", timeout=6)
        if resp.status_code == 404 or not resp.text.strip():
            return {"status": "erro", "mensagem": f"Produto {produto_id} não existe no catálogo."}
        resp.raise_for_status()
        produto = resp.json()

        carrinho = _ler_carrinho()
        item = next((i for i in carrinho if i.id == produto_id), None)
        if item:
            item.quantidade += quantidade
        else:
            carrinho.append(ItemCarrinho(
                id=produto["id"],
                titulo=produto["title"],
                preco_unitario_usd=round(produto["price"], 2),
                quantidade=quantidade,
            ))

        _salvar_carrinho(carrinho)
        return {
            "status": "sucesso",
            "mensagem": f"Adicionado: {quantidade}x '{produto['title']}'.",
            "itens_no_carrinho": len(carrinho),
        }
    except Exception as e:
        return {"status": "erro", "mensagem": f"Erro ao adicionar produto: {e}"}


@tool
def ver_carrinho() -> dict:
    """Mostra os itens do carrinho, o total em dólares e o total em reais (cotação atual)."""
    carrinho = _ler_carrinho()
    if not carrinho:
        return {"status": "vazio", "mensagem": "O carrinho está vazio."}

    total_usd = round(sum(i.subtotal_usd for i in carrinho), 2)
    resultado = {
        "status": "sucesso",
        "total_itens": sum(i.quantidade for i in carrinho),
        "total_usd": total_usd,
        "itens": [i.model_dump() for i in carrinho],
    }

    try:
        taxa = _obter_cotacao_brl()
        resultado["total_brl"] = round(total_usd * taxa, 2)
        resultado["taxa_cambio"] = taxa
    except Exception:
        resultado["total_brl"] = None  # câmbio indisponível; o restante continua válido

    return resultado


@tool(args_schema=AtualizarQuantidadeArgs)
def atualizar_quantidade_carrinho(produto_id: int, nova_quantidade: int) -> dict:
    """
    Define a quantidade total de um item que já está no carrinho.
    Com nova_quantidade = 0, o item é removido do carrinho.
    """
    carrinho = _ler_carrinho()
    item = next((i for i in carrinho if i.id == produto_id), None)
    if not item:
        return {"status": "erro", "mensagem": f"Produto {produto_id} não encontrado no carrinho."}

    if nova_quantidade == 0:
        carrinho.remove(item)
        _salvar_carrinho(carrinho)
        return {"status": "sucesso", "mensagem": f"Produto {produto_id} removido do carrinho."}

    item.quantidade = nova_quantidade
    _salvar_carrinho(carrinho)
    return {
        "status": "sucesso",
        "mensagem": f"Quantidade do produto {produto_id} atualizada para {nova_quantidade}.",
        "subtotal_usd": item.subtotal_usd,
    }


@tool(args_schema=ConfirmarArgs)
def limpar_carrinho(confirmar: bool = False) -> dict:
    """
    Remove TODOS os itens do carrinho. Ação destrutiva: pergunte ao usuário
    antes e só chame com confirmar=True após a confirmação explícita.
    """
    if not confirmar:
        return {
            "status": "confirmacao_necessaria",
            "mensagem": "Peça confirmação ao usuário antes de esvaziar o carrinho.",
        }
    _salvar_carrinho([])
    return {"status": "sucesso", "mensagem": "Carrinho esvaziado com sucesso."}


# ==========================================
# 3. PEDIDOS
# ==========================================
@tool
def finalizar_compra() -> dict:
    """Finaliza a compra: registra o carrinho como pedido em data/pedidos.json e esvazia o carrinho."""
    carrinho = _ler_carrinho()
    if not carrinho:
        return {"status": "erro", "mensagem": "O carrinho está vazio."}

    agora = datetime.now()
    pedido = Pedido(
        pedido_id=f"PED-{agora.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}",
        data_hora=agora.isoformat(),
        itens=carrinho,
    )

    pedidos = _ler_pedidos()
    pedidos.append(pedido)
    _salvar_pedidos(pedidos)
    _salvar_carrinho([])

    return {"status": "sucesso", "mensagem": "Pedido registrado com sucesso!", "pedido": pedido.model_dump()}


@tool
def ver_pedidos() -> dict:
    """Mostra todos os pedidos registrados."""
    pedidos = _ler_pedidos()
    if not pedidos:
        return {"status": "vazio", "mensagem": "Nenhum pedido registrado."}

    return {
        "status": "sucesso",
        "total_pedidos": len(pedidos),
        "pedidos": [p.model_dump() for p in pedidos],
    }


@tool(args_schema=AtualizarPedidoArgs)
def atualizar_pedido(pedido_id: str, produto_id: int, nova_quantidade: int) -> dict:
    """
    Altera a quantidade de um produto dentro de um pedido já registrado.
    O preço unitário gravado no pedido é mantido e o total é recalculado em código.
    Com nova_quantidade = 0, o produto é removido do pedido.
    """
    pedidos = _ler_pedidos()
    pedido = next((p for p in pedidos if p.pedido_id == pedido_id), None)
    if not pedido:
        return {"status": "erro", "mensagem": f"Pedido {pedido_id} não encontrado."}

    item = next((i for i in pedido.itens if i.id == produto_id), None)
    if not item:
        return {"status": "erro", "mensagem": f"Produto {produto_id} não está no pedido {pedido_id}."}

    if nova_quantidade == 0:
        pedido.itens.remove(item)
    else:
        item.quantidade = nova_quantidade

    _salvar_pedidos(pedidos)
    return {"status": "sucesso", "mensagem": f"Pedido {pedido_id} atualizado.", "pedido": pedido.model_dump()}


@tool(args_schema=ExcluirPedidoArgs)
def excluir_pedido(pedido_id: str, confirmar: bool = False) -> dict:
    """
    Exclui um pedido pelo ID. Ação destrutiva: pergunte ao usuário antes e só
    chame com confirmar=True após a confirmação explícita.
    """
    pedidos = _ler_pedidos()
    pedido = next((p for p in pedidos if p.pedido_id == pedido_id), None)
    if not pedido:
        return {"status": "erro", "mensagem": f"Pedido {pedido_id} não encontrado."}

    if not confirmar:
        return {
            "status": "confirmacao_necessaria",
            "mensagem": f"Peça confirmação ao usuário antes de excluir o pedido {pedido_id}.",
        }

    pedidos.remove(pedido)
    _salvar_pedidos(pedidos)
    return {"status": "sucesso", "mensagem": f"Pedido {pedido_id} excluído com sucesso."}


@tool(args_schema=ConfirmarArgs)
def limpar_pedidos(confirmar: bool = False) -> dict:
    """
    Remove TODOS os pedidos registrados. Ação destrutiva e irreversível: pergunte
    ao usuário antes e só chame com confirmar=True após a confirmação explícita.
    """
    if not confirmar:
        return {
            "status": "confirmacao_necessaria",
            "mensagem": "Peça confirmação ao usuário antes de apagar todos os pedidos.",
        }
    _salvar_pedidos([])
    return {"status": "sucesso", "mensagem": "Todos os pedidos foram removidos com sucesso."}


# ==========================================
# 4. CÂMBIO
# ==========================================
@tool(args_schema=ConverterPrecoArgs)
def converter_preco_para_brl(preco_usd: float) -> dict:
    """
    Converte um valor em dólares para reais usando a cotação atual (com cache de 10 min).
    Para o total do carrinho, prefira ver_carrinho, que já retorna total_brl.
    """
    try:
        taxa = _obter_cotacao_brl()
        return {
            "status": "sucesso",
            "preco_usd": preco_usd,
            "preco_brl": round(preco_usd * taxa, 2),
            "taxa_cambio": taxa,
        }
    except Exception as e:
        return {"status": "erro", "mensagem": f"Falha ao consultar a API de câmbio: {e}"}


# Lista pronta para passar ao agente (ex.: create_agent(model, tools=TOOLS))
TOOLS = [
    buscar_produtos_catalogo,
    adicionar_ao_carrinho,
    ver_carrinho,
    atualizar_quantidade_carrinho,
    limpar_carrinho,
    finalizar_compra,
    ver_pedidos,
    atualizar_pedido,
    excluir_pedido,
    limpar_pedidos,
    converter_preco_para_brl,
]