from typing import Optional
from langchain_core.tools import tool
from langchain_core.runnables import Runnable

# Global reference to the retriever runnable
_retriever: Optional[Runnable] = None

def set_retriever(retriever: Runnable):
    """Sets the retriever instance to be used by the search_laws tool."""
    global _retriever
    _retriever = retriever

@tool
def search_laws(query: str) -> str:
    """
    Поиск по базе знаний (законы РФ, кодексы, судебная практика).

    Используй этот инструмент, когда нужно найти юридическую информацию, статьи законов,
    разъяснения или судебные прецеденты для ответа на вопрос пользователя.

    Args:
        query: Строка запроса для поиска (например, "неустойка по ДДУ 214-ФЗ" или "статья 20 ЗоЗПП").

    Returns:
        Строка с найденными фрагментами документов.
    """
    if _retriever is None:
        return "Ошибка: Инструмент поиска не настроен (retriever is None)."

    try:
        # The retriever returns a list of Documents. We need to format them into a string.
        docs = _retriever.invoke(query)
        if not docs:
            return "По вашему запросу ничего не найдено."

        results = []
        for i, doc in enumerate(docs):
            content = doc.page_content
            source = doc.metadata.get("source", "Unknown")

            # Enhance source with Chapter/Article info
            chapter = doc.metadata.get("chapter")
            article = doc.metadata.get("article")

            source_parts = [source]
            if chapter:
                source_parts.append(chapter)
            if article:
                source_parts.append(article)

            full_source = ", ".join(source_parts)

            title = doc.metadata.get("title", "")
            results.append(f"--- Документ {i+1} ---\nИсточник: {full_source} {title}\nТекст: {content}")

        return "\n\n".join(results)
    except Exception as e:
        return f"Ошибка при выполнении поиска: {str(e)}"

@tool
def calculate_penalty_214fz(price: float, delay_days: int) -> str:
    """
    Калькулятор неустойки по 214-ФЗ (Долевое строительство).

    Используй этот инструмент, если вопрос касается задержки сдачи квартиры,
    нарушения сроков по ДДУ (Договор долевого участия).

    Формула: (Цена договора * Ставка рефинансирования ЦБ РФ / 300) * Дни просрочки.
    Текущая ставка рефинансирования: 21%.

    Args:
        price: Стоимость квартиры (объекта долевого строительства) в рублях.
        delay_days: Количество дней просрочки.

    Returns:
        Строка с расчетом неустойки.
    """
    # Hardcoded refinancing rate as per instructions
    rate = 21.0

    # Formula: price * delay * (rate / 100) / 300
    # Note: For individuals (citizens), the penalty is usually double (1/150),
    # but this tool calculates the base 1/300 rate as requested.
    daily_penalty = (price * (rate / 100)) / 300
    total_penalty = daily_penalty * delay_days

    return (
        f"Расчет неустойки по 214-ФЗ:\n"
        f"- Цена объекта: {price:,.2f} руб.\n"
        f"- Дней просрочки: {delay_days}\n"
        f"- Ставка ЦБ РФ: {rate}%\n"
        f"- Формула: (Цена * Ставка / 300) * Дни\n"
        f"- Размер неустойки (1/300 ставки): {total_penalty:,.2f} руб.\n"
        f"(Примечание: Для физических лиц суд часто взыскивает неустойку в двойном размере — 1/150 ставки, "
        f"то есть {total_penalty * 2:,.2f} руб.)"
    )

@tool
def calculate_penalty_zpp(price: float, delay_days: int) -> str:
    """
    Калькулятор неустойки по Закону о защите прав потребителей (ЗоЗПП).

    Используй этот инструмент для расчета неустойки за нарушение сроков выполнения работ/услуг
    или передачи товара (не по ДДУ). Обычно это 1% (товар) или 3% (услуга/работы) в день.

    Этот инструмент считает по ставке 1% (для товаров) и 3% (для услуг/работ).

    Args:
        price: Стоимость товара или услуги/работ в рублях.
        delay_days: Количество дней просрочки.

    Returns:
        Строка с расчетом неустойки.
    """
    penalty_1_percent = price * 0.01 * delay_days
    penalty_3_percent = price * 0.03 * delay_days

    # Cap usually applies (cannot exceed total price), but we'll return the raw calculation
    # and let the LLM explain the caps if found in laws.

    return (
        f"Расчет неустойки по ЗоЗПП (Закон о защите прав потребителей):\n"
        f"- Цена: {price:,.2f} руб.\n"
        f"- Дней просрочки: {delay_days}\n\n"
        f"Вариант 1 (Товары - 1% в день, ст. 23 ЗоЗПП):\n"
        f"- Сумма: {penalty_1_percent:,.2f} руб.\n\n"
        f"Вариант 2 (Работы/Услуги - 3% в день, ст. 28 ЗоЗПП):\n"
        f"- Сумма: {penalty_3_percent:,.2f} руб.\n"
        f"(Примечание: Сумма неустойки не может превышать стоимость товара/работы/услуги)."
    )
