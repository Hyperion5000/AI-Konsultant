import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from bot.graph.tools import (
    calculate_penalty_214fz,
    calculate_penalty_zpp,
    search_laws,
    set_retriever
)

def test_calculate_penalty_214fz():
    result = calculate_penalty_214fz.invoke({"price": 10_000_000, "delay_days": 5})
    assert "35,000.00 руб." in result
    assert "214-ФЗ" in result

def test_calculate_penalty_zpp():
    result = calculate_penalty_zpp.invoke({"price": 100_000, "delay_days": 5})
    assert "5,000.00 руб." in result
    assert "15,000.00 руб." in result
    assert "ЗоЗПП" in result

def test_search_laws_no_retriever():
    set_retriever(None)
    result = search_laws.invoke({"query": "test"})
    assert "Ошибка: Инструмент поиска не настроен" in result

def test_search_laws_success():
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="Content 1", metadata={"source": "Source 1"}),
        Document(page_content="Content 2", metadata={"source": "Source 2", "title": "Title 2"})
    ]
    set_retriever(mock_retriever)

    result = search_laws.invoke({"query": "test"})
    assert "--- Документ 1 ---" in result
    assert "Content 1" in result

def test_search_laws_with_metadata():
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(
            page_content="Content 3",
            metadata={
                "source": "Law.txt",
                "chapter": "Глава 1",
                "article": "Статья 5"
            }
        )
    ]
    set_retriever(mock_retriever)

    result = search_laws.invoke({"query": "metadata test"})
    # Expected: "Источник: Law.txt, Глава 1, Статья 5 "
    assert "Law.txt, Глава 1, Статья 5" in result

def test_search_laws_empty():
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    set_retriever(mock_retriever)

    result = search_laws.invoke({"query": "test"})
    assert "По вашему запросу ничего не найдено" in result

def test_search_laws_exception():
    mock_retriever = MagicMock()
    mock_retriever.invoke.side_effect = Exception("Search failed")
    set_retriever(mock_retriever)

    result = search_laws.invoke({"query": "test"})
    assert "Ошибка при выполнении поиска: Search failed" in result
