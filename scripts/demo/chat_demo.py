import os
import sys
import asyncio
from colorama import Fore, Style, init
from openai import AsyncClient

sys.path.append(os.getcwd())

from config.setting import settings

init(autoreset=True)

client = AsyncClient(api_key=settings.llm.api_key, base_url=settings.llm.api_base)


async def chat_loop():
    # 用于存储上下文的 messages
    messages = [{"role": "system", "content": "你是一个乐于助人的 AI 助手"}]

    while True:
        user_input = input(f"{Fore.GREEN}用户: {Style.RESET_ALL}")

        if user_input.lower() in {"exit", "quit"}:
            print("退出聊天。再见！")
            break

        messages.append({"role": "user", "content": user_input})
        print(f"{Fore.YELLOW}AI 正在思考...{Style.RESET_ALL}")

        full_response = ""

        try:
            stream = await client.chat.completions.create(
                model="qwen-max", messages=messages, stream=True, temperature=0.7
            )

            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        print(delta, end="", flush=True)
                        full_response += delta

            print()  # 换行

            messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            print(f"{Fore.RED}发生错误: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        print("\n程序已终止")
