import os
import sys

sys.path.append(os.getcwd())

from src.app.agents import invoke


if __name__ == "__main__":
    result = invoke("2024 年泸州雨露计划补贴标准是多少？")
    print(result.get("final_response"))