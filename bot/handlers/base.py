from aiogram import Router, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from bot.rag_service import RAGService

router = Router()

@router.message(CommandStart())
async def command_start_handler(message: Message):
    """
    Handler for `/start` command.
    """
    await message.answer(
        f"Привет, {message.from_user.full_name}!\n"
        "Я — твой юридический ИИ-консультант.\n"
        "Задай мне вопрос, и я отвечу на основе законов РФ.\n\n"
        "Команды:\n"
        "/reset - начать новый диалог (очистить контекст)\n"
        "/help - справка"
    )

@router.message(Command("help"))
async def command_help_handler(message: Message):
    """
    Handler for `/help` command.
    """
    await message.answer(
        "Я использую базу знаний (индексированные документы) для поиска ответов.\n"
        "Также я помню последние несколько сообщений нашего диалога.\n"
        "Если вы хотите сменить тему или чтобы я забыл предыдущий контекст, используйте команду /reset.\n"
        "Ответ может занимать время, так как я работаю на локальном оборудовании."
    )

@router.message(Command("reset"))
async def command_reset_handler(message: Message, rag_service: RAGService):
    """
    Handler for `/reset` command.
    """
    if rag_service:
        rag_service.reset_history(message.from_user.id)
        await message.answer("История диалога очищена. Можем начать сначала!")
    else:
        await message.answer("Сервис еще не готов.")
