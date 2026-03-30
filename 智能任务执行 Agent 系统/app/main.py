import argparse
import asyncio

from agent import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP 编排器")
    parser.add_argument("prompt", help="发送给 LLM 的提示词")
    args = parser.parse_args()
    print(asyncio.run(run(args.prompt)) or "")
