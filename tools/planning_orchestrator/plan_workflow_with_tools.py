from __future__ import annotations
import json, re
from typing import Any
from fastmcp import Context, Client
from resources.semantic_model.utils import get_model
from .prompts import system_prompt_orchestrator

# -------- Planner tool that can CALL planning tools via an inner loop ----------
async def plan_workflow_with_tools(
    user: str = "",
    name: str = "",
    user_question: str = "",
    ctx: Context = None,
    # Optional: allow the caller to pass a precomputed executor tool catalog (name+desc)
    allowed_executor_tools: list[str] = None,
    # Observations from prior execution steps, to re-plan mid-flight
    observations: list[dict[str, Any]] = None,
    # Budget and safety
    max_steps: int = 4,
) -> dict:
    """Tool for generating a step-by-step plan to answer a user's question using available planning and executor tools."""
    # 1) Discover tools and split them by tag using an in-memory FastMCP client.
    #    This keeps metadata/tags consistent with what clients see.  (Client(server) is supported)
    #    NOTE: tags are present under meta._fastmcp.tags in tool listings.
    # ------------------------------------------------------------------------
    print("\n[DEBUG] Here:\n", user_question)
    
    print(f"Client class: {Client!r}")  # should show fastmcp.Client
    server = ctx.fastmcp
    print(f"Server type: {type(server)}")

    
    try: 
        async with Client(server) as local_client:
            all_tools = await local_client.list_tools()
            print("\n[DEBUG] list tools:\n", all_tools)
    except Exception as e:
        print("\n[ERROR] listing tools failed:\n", str(e))

    planning_tools = [t for t in all_tools if t.name in ("get_style_guide",)]
    executor_tools = (
        [
            {"name": t.name, "description": t.description}
            for t in all_tools if t.name in allowed_executor_tools
        ]
    )
    print("\n[DEBUG] list executor tools:", executor_tools)

    # For the LLM, give a small, readable catalog
    planning_tools_visible = [
        {"name": t.name, "description": t.description} for t in planning_tools
    ]

    # Handy index for allow-list enforcement
    planning_allow = {t["name"] for t in planning_tools_visible}

    # 2) Build system + initial user messages for the planner agent
    try: 
        model = get_model(user, name) if user and name else {}
        if model:
            format = "ttl/owl" if "ttl" in model.keys() else "xmi/uml"
    except Exception as e:
        print("\n[ERROR] get_model failed:\n", str(e))
        model = {}

    user_block = {
        "user_question": user_question,
        "user_info": {
            "user": user if user else "anonymous",
            "name": name if name else "default",
            # Potentially useful metadata about the data model
            "provided_data_model": "yes" if model else "no",
            "data_model_format": format if model else "unknown",
            },
        "observations": observations or [],
        "planning_tools_you_can_call": planning_tools_visible,
        "executor_tools_for_final_plan": executor_tools,
    }
    #print("\n[DEBUG] User Block:\n", user_block)

    messages = [
        json.dumps(user_block, ensure_ascii=False, indent=2)
    ]

    scratch: list[dict] = []  # keep thought/action/observation triplets for transparency

    # 3) Agent loop: sample -> (action or final_plan)

    step = 0
    while step < max_steps:
        sample = await ctx.sample(
            messages=messages,
            system_prompt=system_prompt_orchestrator,
            temperature=0.0,
            max_tokens=800,
        )
        print("\n[DEBUG] Sample:\n", str(sample))
        text = getattr(sample, "text", str(sample))

        m = re.search(r"\{.*\}", text, re.S)
        try:
            obj = json.loads(m.group(0) if m else text)
        except Exception:
            messages.append(f"Please return VALID JSON. Your last output was:\n{text[:2000]}")
            step += 1
            continue

        # If finalize
        if "final_plan" in obj:
            plan = obj["final_plan"]
            plan.setdefault("plan_steps", [])
            plan.setdefault("tools_to_call", [])
            plan.setdefault("resources_used", [])
            plan.setdefault("notes", "")

            # Check for planning-only tools in tools_to_call
            planning_only_calls = [t for t in plan["tools_to_call"] if t["tool"] not in allowed_executor_tools]
            if planning_only_calls:
                # Call each planning-only tool, add its observation, and re-plan
                for call in planning_only_calls:
                    tool_name = call["tool"]
                    args = call.get("args_template", {})
                    print(f"[DEBUG] Calling planning-only tool: {tool_name} with args: {args}")
                    async with Client(server) as local_client:
                        result = await local_client.call_tool(tool_name, args)
                        obs = result.data if getattr(result, "data", None) is not None else (
                            "".join(block.text for block in (result.content or []) if hasattr(block, "text"))
                        )
                    scratch.append({"step": step + 1, "tool": tool_name, "args": args, "observation": obs})
                    messages.append(json.dumps({"observation": obs}, ensure_ascii=False))
                # Remove planning-only tools from tools_to_call and re-plan
                step += 1
                continue

            # attach scratch for auditability
            plan["debug_trace"] = scratch
            return plan

        # If action
        action = obj.get("action")
        if action:
            tool_name = action.get("tool")
            args = action.get("args", {}) or {}

            if tool_name not in planning_allow:
                scratch.append({
                    "step": step + 1,
                    "tool": tool_name,
                    "args": args,
                    "observation": f"DENIED: tool '{tool_name}' is not in planner allow-list."
                })
                messages.append(json.dumps({"observation": scratch[-1]["observation"]}))
                step += 1
                continue

            async with Client(server) as local_client:
                result = await local_client.call_tool(tool_name, args)
                obs = result.data if getattr(result, "data", None) is not None else (
                    "".join(block.text for block in (result.content or []) if hasattr(block, "text"))
                )
            scratch.append({"step": step + 1, "tool": tool_name, "args": args, "observation": obs})
            messages.append(json.dumps({"observation": obs}, ensure_ascii=False))
            step += 1
            continue
        messages.append("Return JSON with either {'action': {...}} or {'final_plan': {...}} only.")
        step += 1

    # 4) Fallback after budget: return best-effort plan
    return {
        "plan_steps": ["No finalization within step budget; returning partial notes."],
        "tools_to_call": [],
        "resources_used": [],
        "notes": "Planner ran out of steps.",
        "debug_trace": scratch,
    }


plan_workflow_with_tools.__doc__ = f"""
    Tool for generating a step-by-step plan to answer a user's question using available planning and executor tools.
    The planner agent may call planning tools to discover or retrieve relevant information, and produces a structured plan
    for a separate executor agent, which is restricted to the allowed executor tools.

    Args:
        user (str, optional):
            Identifier passed to `get_model` to locate the model. Defaults to "".
        name (str, optional):
            Model name passed to `get_model` to locate the model. Defaults to "".
        user_question (str):
            The user's original question to be answered by the plan.

        observations (Optional[list[dict[str, Any]]]):
            Observations from prior execution steps, to re-plan mid-flight.
            Defaults to None (if no prior observations).
    
        max_steps (int):
            The maximum number of planning steps (LLM calls) to perform to arrive at a final plan.
            Defaults to 4.

    Returns:
        dict:
            A dictionary containing the following keys:
                - plan_steps (list of str): The ordered steps of the proposed plan.
                - tools_to_call (list of dict): Each dict contains the name and arguments for executor tools to be called.
                - resources_used (list of str): Any resources or references used during planning.
                - notes (str): Additional notes or rationale for the plan.
                - debug_trace (list of dict): Internal trace of planning steps, actions, and observations for auditability.
            If the planner does not finalize within the step budget, a partial plan is returned with notes.
"""
