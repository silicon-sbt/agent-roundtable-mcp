"""Roundtable-owned LLM object protocol and deterministic mock.

Provider transports, credentials, model normalization and retries belong to
``llm_gateway``. This module keeps only the object contract required by the
roundtable graph plus its offline demo implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    """Minimal object contract consumed by roundtable nodes."""

    provider_name: str
    model: str

    def generate(self, prompt: str) -> str: ...


def _extract(prompt: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}：(.+)", prompt)
    return match.group(1).strip() if match else ""


@dataclass
class MockLLM:
    """Deterministic local LLM for tests and no-key demos."""

    model: str = "mock"
    provider_name: str = "mock"

    def generate(self, prompt: str) -> str:
        topic = _extract(prompt, "话题") or "这个话题"
        if "[ROUND_TABLE_MODERATOR_QUESTION]" in prompt:
            round_text = _extract(prompt, "当前轮次") or "本轮"
            return (
                f"{round_text} 我们先把问题收窄：围绕“{topic}”，请各位说明最关键的判断依据，"
                "并直接回应上一位的逻辑漏洞或盲点。"
            )
        if "[ROUND_TABLE_ROUND_SUMMARY]" in prompt:
            return (
                "1. 本轮把讨论从直觉判断推进到因果链拆解。\n"
                "2. 主要争议在于短期冲击和长期适应能力谁更重要。\n"
                "3. 下一轮应继续追问哪些假设最容易被现实推翻。"
            )
        if "[ROUND_TABLE_FINAL_SUMMARY]" in prompt:
            return (
                "## 主要共识\n"
                "- 这个问题不能用单一变量解释，需要同时看技术、制度、周期和人性。\n\n"
                "## 主要分歧\n"
                "- 分歧集中在风险出现的速度、可控性，以及普通人能否及时调整。\n\n"
                "## 最有价值观点\n"
                "- 最值得保留的是把宏大判断拆成可验证假设，而不是只做情绪化站队。\n\n"
                "## 风险提示\n"
                "- 涉及现实经济或市场数据的判断需要联网核实，不能把模拟讨论当成事实来源。\n\n"
                "## 最终总结\n"
                f"- 圆桌倾向于认为“{topic}”存在真实变量和不确定性，行动上应保持开放、审慎和可调整。"
            )

        persona_name = _extract(prompt, "- 名称") or "Agent"
        role = _extract(prompt, "- 角色") or "观察者"
        catchphrases = _extract(prompt, "- 口头禅")
        phrase = catchphrases.split(",")[0].strip() if catchphrases else "先拆开看"
        return (
            f"{phrase}。我作为{role}，会先回应上一位：他的观点有价值，但容易把“{topic}”看成单线故事。"
            "更稳妥的做法是把它拆成几个可检验假设：谁受益、谁承担成本、变化速度有多快、制度能否缓冲。"
            "如果有本地参考资料，我会把它当作背景线索而不是绝对答案；如果涉及现实经济数据，需要联网核实。"
            "在没有核实时，我只讨论逻辑结构而不报具体数字。"
            f"这是 {persona_name} 的基本判断。"
        )


def describe_llm(llm: LLMClient) -> dict[str, str]:
    """Return the effective identity/model exposed by a roundtable LLM object."""

    return {
        "provider": str(getattr(llm, "provider_name", llm.__class__.__name__)),
        "model": str(getattr(llm, "model", "unknown")),
    }


__all__ = ["LLMClient", "MockLLM", "describe_llm"]
