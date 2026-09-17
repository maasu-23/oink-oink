"""Bedrock Converse API tool-use schemas for the supervisor tools in tools.py.

Kept separate from tools.py so the AWS-facing JSON schema shape doesn't leak
into the plain-Python tool implementations.
"""

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "list_training_runs",
                "description": "List all training runs (variant/seed combinations) that have logged metrics on disk.",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }
        },
        {
            "toolSpec": {
                "name": "get_training_metrics",
                "description": "Get reward/episode statistics for one training run, e.g. final reward, best/worst episode, total timesteps.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "run_name": {"type": "string", "description": "e.g. 'connectome_seed0'. Use list_training_runs to see valid names."},
                            "tail_episodes": {"type": "integer", "description": "Number of most-recent episodes to average for the final reward (default 20)."},
                        },
                        "required": ["run_name"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "compare_variants",
                "description": "Compare final performance across all seeds between two network variants (e.g. connectome vs. baseline).",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "variant_a": {"type": "string"},
                            "variant_b": {"type": "string"},
                        },
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "get_circuit_summary",
                "description": "Get the extracted fly connectome circuit's neuron counts and edge counts per layer (input/processing/output), plus a short biology description.",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }
        },
        {
            "toolSpec": {
                "name": "list_neurons",
                "description": "List sample FlyWire root_ids for a given neuron role (input/processing/output). Use this to discover real neuron ids before calling get_neuron_connections.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string", "enum": ["input", "processing", "output"]},
                            "limit": {"type": "integer", "description": "Max ids to return (default 10)."},
                        },
                        "required": ["role"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "get_neuron_connections",
                "description": "Look up a specific neuron's synaptic partners (upstream and/or downstream) by its FlyWire root_id, with synapse weights.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "root_id": {"type": "integer"},
                            "direction": {"type": "string", "enum": ["upstream", "downstream", "both"]},
                            "limit": {"type": "integer", "description": "Max partners to return per direction (default 20)."},
                        },
                        "required": ["root_id"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "trigger_retrain",
                "description": "Kick off a new PPO training run in the background for a given network variant and seed. Returns immediately; does not wait for training to finish.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "variant": {"type": "string", "enum": ["connectome", "baseline"]},
                            "seed": {"type": "integer"},
                            "timesteps": {"type": "integer", "description": "Default 150000."},
                        },
                        "required": ["variant", "seed"],
                    }
                },
            }
        },
    ]
}
