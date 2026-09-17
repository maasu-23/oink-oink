"""
Phase 4 — Bedrock training supervisor.

This is an automated training *supervisor*: it answers questions about
training runs and the connectome by calling tools that read real data
(monitor logs, the extracted adjacency matrix) and can trigger a retrain.
It does not learn or improve itself over time — only the RL agent it
supervises does that. See PLAN.md's "Honesty note for writeups."
"""
import json
import os

import boto3

from oink.supervisor.tool_schemas import TOOL_CONFIG
from oink.supervisor.tools import TOOL_REGISTRY

# Set BEDROCK_MODEL_ID to your inference profile ARN (from the Bedrock
# console/playground's "View API request") -- not hardcoded here since it
# embeds your AWS account ID. See README/.env.example.
DEFAULT_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID")
DEFAULT_REGION = os.environ.get("AWS_REGION", "ap-south-1")

SYSTEM_PROMPT = """\
You are the training supervisor for PigBrain, a project comparing a fly-connectome-constrained \
reinforcement learning network against a size-matched random baseline on a pig-dodge task. \
You have tools to read real training metrics and the extracted connectome graph, and to trigger \
new training runs. Always call a tool to get real numbers before answering questions about \
training results or the circuit -- never guess or fabricate metrics. When asked to explain why \
a network behaves a certain way, ground your answer in the actual synapse data from \
get_neuron_connections or get_circuit_summary where relevant. Be concise and precise; state \
uncertainty when the data doesn't fully support a claim.
"""


class Supervisor:
    def __init__(self, model_id: str | None = DEFAULT_MODEL_ID, region: str = DEFAULT_REGION):
        if not model_id:
            raise ValueError(
                "No model id set. Export BEDROCK_MODEL_ID to your Bedrock inference "
                "profile ARN (see .env.example) or pass model_id explicitly."
            )
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.messages = []

    def _run_tool(self, name: str, tool_input: dict) -> dict:
        fn = TOOL_REGISTRY.get(name)
        if fn is None:
            return {"error": f"unknown tool '{name}'"}
        try:
            return fn(**tool_input)
        except Exception as exc:  # tool errors should be visible to the model, not crash the loop
            return {"error": str(exc)}

    def ask(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": [{"text": user_message}]})

        while True:
            response = self.client.converse(
                modelId=self.model_id,
                system=[{"text": SYSTEM_PROMPT}],
                messages=self.messages,
                toolConfig=TOOL_CONFIG,
            )
            output_message = response["output"]["message"]
            self.messages.append(output_message)

            stop_reason = response["stopReason"]
            if stop_reason != "tool_use":
                return "".join(block.get("text", "") for block in output_message["content"])

            tool_results = []
            for block in output_message["content"]:
                if "toolUse" not in block:
                    continue
                tool_use = block["toolUse"]
                result = self._run_tool(tool_use["name"], tool_use.get("input", {}))
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_use["toolUseId"],
                            "content": [{"json": result}],
                        }
                    }
                )
            self.messages.append({"role": "user", "content": tool_results})


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Chat with the PigBrain training supervisor.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("question", nargs="*", help="Ask a single question and exit; omit for an interactive chat loop.")
    args = parser.parse_args()

    supervisor = Supervisor(model_id=args.model_id, region=args.region)

    if args.question:
        print(supervisor.ask(" ".join(args.question)))
        return

    print("PigBrain training supervisor. Ctrl-D to exit.")
    while True:
        try:
            user_input = input("\nyou> ")
        except EOFError:
            break
        if not user_input.strip():
            continue
        print("\nsupervisor>", supervisor.ask(user_input))


if __name__ == "__main__":
    main()
