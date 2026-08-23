from __future__ import annotations

import argparse

from roundtable.graph import run_roundtable
from llm import API_PROVIDERS, CLI_BACKENDS, create_llm, create_llm_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a persona-based agent roundtable.")
    parser.add_argument("--topic", required=True, help="Discussion topic")
    parser.add_argument("--council", default="experts", help="Council config name")
    parser.add_argument("--rounds", type=int, default=2, help="Number of discussion rounds")
    parser.add_argument(
        "--provider",
        default="auto",
        choices=list(dict.fromkeys((*API_PROVIDERS, *CLI_BACKENDS))),
        help="LLM provider. auto uses persona/council config or falls back to mock.",
    )
    parser.add_argument("--model", default=None, help="Provider model override")
    parser.add_argument(
        "--provider-chain",
        default=None,
        help="Workflow-style fallback chain, for example 'claude:sonnet@high, gemini:gemini-2.5-flash'.",
    )
    parser.add_argument("--mock", action="store_true", help="Force deterministic mock LLM")
    parser.add_argument("--output-dir", default="logs", help="Markdown log output directory")
    parser.add_argument(
        "--agent-llm-config",
        default=None,
        help="JSON file for per-agent LLM config. Defaults to config/agent_llms.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mock:
        llm = create_llm(provider="mock", model=args.model)
    elif args.provider_chain:
        llm = create_llm_from_config({"provider_chain": args.provider_chain})
    elif args.provider in CLI_BACKENDS:
        provider_chain = f"{args.provider}:{args.model}" if args.model else args.provider
        llm = create_llm_from_config({"provider_chain": provider_chain})
    elif args.provider == "auto":
        llm = None
    else:
        llm = create_llm(provider=args.provider, model=args.model)
    result = run_roundtable(
        topic=args.topic,
        council_name=args.council,
        rounds=args.rounds,
        llm=llm,
        output_dir=args.output_dir,
        agent_llm_config_path=args.agent_llm_config,
    )
    print(result["final_summary"])
    print(f"\nLog saved to: {result['log_path']}")


if __name__ == "__main__":
    main()
