"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude Haiku 4.5 - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.

Commands:
  python src/agent_orchestrator.py test            # 3 scenarios, local run, traced to X-Ray
  python src/agent_orchestrator.py chat            # interactive terminal chat
  python src/agent_orchestrator.py deploy          # Tasks 3-6 deployment pipeline
  python src/agent_orchestrator.py invoke "<msg>"  # call the deployed AgentCore Runtime
  python src/agent_orchestrator.py serve           # HTTP server (what AgentCore Runtime runs)
"""

import boto3
import json
import time
import os
import sys
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key

import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)

# ─────────────────────────────────────────────────────
# OBSERVABILITY
# ─────────────────────────────────────────────────────
# `tool` is the Strands @tool decorator wrapped so that every tool call is
# recorded as an X-Ray subsegment (the orchestrator's route_to_* tools become
# the worker-agent nodes on the X-Ray Service Map) and logged at INFO level.
# Use it exactly like `strands.tool`:  @tool  above each tool function.
from agent_observability import (
    tool, tracer, setup_logging, flush_logs, print_trace_hint,
    apply_observability_config, wait_for_runtime_ready,
)


# ─────────────────────────────────────────────────────
# AWS CLIENTS
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)

def _update_workflow_state(
    session_id: str,
    updates: dict,
    expected_version: int,
    max_retries: int = 3
) -> dict:
    """
    Update WorkflowState with optimistic locking.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [
                f"{k} = :{k}" for k in updates
            ]
            update_expr_parts.append("version = :new_version")

            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {
                f":{k}": v
                for k, v in updates.items()
            }

            expr_values[":new_version"] = expected_version + 1
            expr_values[":expected_version"] = expected_version

            table.update_item(
                Key={"session_id": session_id},
                UpdateExpression=update_expr,
                ConditionExpression="version = :expected_version",
                ExpressionAttributeValues=expr_values
            )

            logger.info(
                f"WorkflowState updated successfully: "
                f"session={session_id}, "
                f"version={expected_version}->{expected_version + 1}"
            )

            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )

            current = _read_workflow_state(session_id)

            if not current:
                raise RuntimeError(
                    f"WorkflowState disappeared during update "
                    f"(session: {session_id})"
                )

            previous_version = expected_version
            expected_version = int(current["version"])

            logger.warning(
                f"WorkflowState version conflict: "
                f"session={session_id}, "
                f"expected={previous_version}, "
                f"current={expected_version}, "
                f"retry={attempt + 1}/{max_retries}"
            )

            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError(
        "WorkflowState update: unexpected exit from retry loop"
    )


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────

def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.1,
    )

    system_prompt = """
You are the NovaMart Inventory Agent.

Your responsibility is to retrieve accurate customer and order facts from
DynamoDB. You are a factual data-retrieval specialist.

You MUST:
- Use your tools to retrieve the requested information.
- Report facts exactly as stored in DynamoDB.
- Never invent or assume customer or order information.
- Never make return, refund, eligibility, or policy decisions.
- Never answer policy questions.
- Clearly report when a customer or order cannot be found.
- Return structured, concise information that downstream agents can use.

Your available tools are:
1. check_order_status - retrieve facts about one order.
2. get_customer_tier - retrieve the customer's account tier.
3. list_customer_orders - retrieve all orders for a customer.
"""

    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """
        Look up one order in DynamoDB and report its status, product, dates
        and amount. Reports facts only - it does NOT decide return eligibility.

        Args:
            customer_id: The customer's unique identifier (e.g. CUST-001)
            order_id: The order identifier (e.g. ORD-27176)

        Returns:
            Order record (order_id, status, product_name, order_date, price, ...)
            or a not-found message
        """
        table = dynamodb.Table(config.ORDERS_TABLE)

        response = table.get_item(
            Key={
                'customer_id': customer_id,
                'order_id': order_id,
            }
        )

        item = response.get('Item')

        if not item:
            return {
                'found': False,
                'customer_id': customer_id,
                'order_id': order_id,
                'message': f'Order {order_id} was not found for customer {customer_id}.',
            }

        return {
            'found': True,
            'order': item,
        }

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.
        Standard customers have a 30-day return window; Premium customers have 60 days.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Customer profile including tier and account details
        """
        table = dynamodb.Table(config.CUSTOMERS_TABLE)

        response = table.get_item(
            Key={
                'customer_id': customer_id,
            }
        )

        item = response.get('Item')

        if not item:
            return {
                'found': False,
                'customer_id': customer_id,
                'message': f'Customer {customer_id} was not found.',
            }

        return {
            'found': True,
            'customer': item,
        }

    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            List of all orders with order_id, status, order_date, and amount
        """
        table = dynamodb.Table(config.ORDERS_TABLE)

        response = table.query(
            KeyConditionExpression=Key('customer_id').eq(customer_id)
        )

        orders = response.get('Items', [])

        return {
            'customer_id': customer_id,
            'order_count': len(orders),
            'orders': orders,
        }

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            check_order_status,
            get_customer_tier,
            list_customer_orders,
        ],
    )


# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────

def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.1,
    )

    system_prompt = """
You are the NovaMart Refund Agent.

Your responsibility is to determine return/refund eligibility using facts
already gathered by the Inventory Agent and stored in WorkflowState.

You MUST:
- Always call get_inventory_context first before making an eligibility decision.
- Base decisions only on the inventory facts available in WorkflowState.
- Standard customers have a 30-day return window.
- Premium customers have a 60-day return window.
- Determine eligibility from the customer's tier, order date, order status,
  and the current date.
- Do not invent missing order or customer information.
- Clearly explain the factual reason for an eligibility decision.
- Only initiate a refund when the customer/order is eligible and the customer
  is actually requesting a return/refund.
- Use initiate_refund only after determining eligibility.
- Do not answer general policy questions; those belong to the Policy Agent.

The Inventory Agent gathers facts. You make the eligibility decision.
"""

    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier

        Returns:
            The inventory_agent field from WorkflowState, or empty dict if not yet set
        """
        state = _read_workflow_state(session_id)

        if not state:
            return {
                'found': False,
                'session_id': session_id,
                'message': 'WorkflowState was not found for this session.',
                'inventory_agent': {},
            }

        return {
            'found': True,
            'session_id': session_id,
            'inventory_agent': state.get('inventory_agent', {}),
        }

    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.

        Args:
            customer_id: The customer's unique identifier
            order_id: The order to return
            reason: Customer-provided reason for the return

        Returns:
            Confirmation dict with return_reference number and instructions
        """
        # Read the workflow state associated with this customer/session before
        # modifying the order. The Refund Agent itself is responsible for the
        # eligibility decision, so this tool only performs the requested update.
        table = dynamodb.Table(config.ORDERS_TABLE)

        response = table.get_item(
            Key={
                'customer_id': customer_id,
                'order_id': order_id,
            }
        )

        item = response.get('Item')

        if not item:
            return {
                'success': False,
                'customer_id': customer_id,
                'order_id': order_id,
                'message': f'Order {order_id} was not found for customer {customer_id}.',
            }

        return_reference = f"RET-{uuid.uuid4().hex[:8].upper()}"

        table.update_item(
            Key={
                'customer_id': customer_id,
                'order_id': order_id,
            },
            UpdateExpression=(
                'SET return_reference = :return_reference, '
                '#status = :status, return_reason = :reason'
            ),
            ExpressionAttributeNames={
                '#status': 'status',
            },
            ExpressionAttributeValues={
                ':return_reference': return_reference,
                ':status': 'RETURN_REQUESTED',
                ':reason': reason,
            },
        )

        return {
            'success': True,
            'customer_id': customer_id,
            'order_id': order_id,
            'return_reference': return_reference,
            'status': 'RETURN_REQUESTED',
            'instructions': (
                'Your return request has been initiated. '
                'Keep the return reference for future communication.'
            ),
        }

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            get_inventory_context,
            initiate_refund,
        ],
    )

# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────
def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    PARALLEL, each querying its own Knowledge Base. The coordinator synthesizes
    the combined results into a complete, grounded policy answer.
    """

    # ─────────────────────────────────────────────────────────────
    # Returns Policy Retriever
    # ─────────────────────────────────────────────────────────────

    @tool
    def retrieve_returns_policy(query: str) -> str:
        """Retrieve relevant passages from the Returns Policy knowledge base."""
        results = retrieve_from_knowledge_base(
            query=query,
            knowledge_base_id=config.RETURNS_KB_ID,
        )
        return format_kb_results(results)

    returns_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.0,
    )

    returns_retriever = Agent(
        name="ReturnsPolicyRetrieverAgent",
        model=returns_model,
        system_prompt="""
You are the NovaMart Returns Policy Retriever Agent.

Your only responsibility is retrieving relevant information from the
Returns Policy knowledge base.

Use retrieve_returns_policy to search the knowledge base.
Return only information grounded in the retrieved passages.
Do not invent policy details.
Do not make unsupported assumptions.
Do not make eligibility decisions.
""",
        tools=[retrieve_returns_policy],
    )

    # ─────────────────────────────────────────────────────────────
    # Shipping Policy Retriever
    # ─────────────────────────────────────────────────────────────

    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """Retrieve relevant passages from the Shipping Policy knowledge base."""
        results = retrieve_from_knowledge_base(
            query=query,
            knowledge_base_id=config.SHIPPING_KB_ID,
        )
        return format_kb_results(results)

    shipping_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.0,
    )

    shipping_retriever = Agent(
        name="ShippingPolicyRetrieverAgent",
        model=shipping_model,
        system_prompt="""
You are the NovaMart Shipping Policy Retriever Agent.

Your only responsibility is retrieving relevant information from the
Shipping Policy knowledge base.

Use retrieve_shipping_policy to search the knowledge base.
Return only information grounded in the retrieved passages.
Do not invent policy details.
Do not make unsupported assumptions.
""",
        tools=[retrieve_shipping_policy],
    )

    # ─────────────────────────────────────────────────────────────
    # Warranty Policy Retriever
    # ─────────────────────────────────────────────────────────────

    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """Retrieve relevant passages from the Warranty Policy knowledge base."""
        results = retrieve_from_knowledge_base(
            query=query,
            knowledge_base_id=config.WARRANTY_KB_ID,
        )
        return format_kb_results(results)

    warranty_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.0,
    )

    warranty_retriever = Agent(
        name="WarrantyPolicyRetrieverAgent",
        model=warranty_model,
        system_prompt="""
You are the NovaMart Warranty Policy Retriever Agent.

Your only responsibility is retrieving relevant information from the
Warranty Policy knowledge base.

Use retrieve_warranty_policy to search the knowledge base.
Return only information grounded in the retrieved passages.
Do not invent policy details.
Do not make unsupported assumptions.
""",
        tools=[retrieve_warranty_policy],
    )

    # ─────────────────────────────────────────────────────────────
    # Parallel Multi-Agent RAG
    # ─────────────────────────────────────────────────────────────

    @tool
    def search_all_policies(query: str) -> str:
        """
        Query all three policy knowledge bases IN PARALLEL and return combined results.

        Runs ReturnsPolicyRetrieverAgent, ShippingPolicyRetrieverAgent, and
        WarrantyPolicyRetrieverAgent simultaneously, then combines their findings.

        Args:
            query: The customer's policy question

        Returns:
            Combined policy passages from all three knowledge bases
        """

        retrievers = {
            'Returns': returns_retriever,
            'Shipping': shipping_retriever,
            'Warranty': warranty_retriever,
        }

        trace.kb_start({
            'Returns': config.RETURNS_KB_ID,
            'Shipping': config.SHIPPING_KB_ID,
            'Warranty': config.WARRANTY_KB_ID,
        })

        def _run_retriever(domain: str, agent, query: str) -> tuple:
            """
            Run one retriever sub-agent and return (domain, result_text).
            """
            try:
                result = agent(query)

                # Strands Agent results can be string-like or expose a
                # response/result representation depending on SDK version.
                result_text = str(result)

                return domain, result_text

            except Exception as exc:
                logging.exception(
                    "Policy retriever failed for %s",
                    domain,
                )
                return domain, f"[{domain} retrieval failed: {exc}]"

        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    _run_retriever,
                    domain,
                    agent,
                    query,
                ): domain
                for domain, agent in retrievers.items()
            }

            for future in as_completed(futures):
                domain, result_text = future.result()
                results[domain] = result_text

        trace.kb_done(len(retrievers))

        for domain in ['Returns', 'Shipping', 'Warranty']:
            trace.kb_result(
                domain,
                results.get(domain, '[No results]')
            )

        combined = []

        for domain in ['Returns', 'Shipping', 'Warranty']:
            combined.append(
                f"=== {domain} Policy ===\n"
                f"{results.get(domain, '[No results]')}"
            )

        return "\n\n".join(combined)

    # ─────────────────────────────────────────────────────────────
    # Policy Coordinator
    # ─────────────────────────────────────────────────────────────

    coordinator_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.2,
    )

    coordinator_system_prompt = """
You are the NovaMart Policy Agent.

You are a policy-answering coordinator responsible for answering customer
questions about returns, shipping, and warranties.

For every policy question:

1. You MUST call search_all_policies first.
2. search_all_policies queries all three specialized policy retrievers
   in parallel.
3. Use the returned knowledge-base passages as the source of truth.
4. Synthesize a concise, accurate answer from the retrieved information.
5. Do not invent policy details.
6. If the retrieved policies do not contain enough information, clearly say
   that the available policy information is insufficient.
7. Distinguish between returns, shipping, and warranty policies when useful.
8. Do not make customer-specific eligibility decisions based only on general
   policy information. Customer-specific eligibility belongs to the Refund
   Agent after Inventory information has been gathered.

Your final response should be grounded in the retrieved policy evidence.
"""

    return Agent(
        name="PolicyAgent",
        model=coordinator_model,
        system_prompt=coordinator_system_prompt,
        tools=[search_all_policies],
    )
# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────
def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Drafts the final customer-facing message by reading the full WorkflowState
    and composing a coherent, empathetic response.
    """

    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.3,
    )

    system_prompt = """
You are the NovaMart Communication Agent.

Your responsibility is to produce the final customer-facing response for
NovaMart customer support.

You MUST:
1. Read the complete WorkflowState using get_full_workflow_context before
   composing the response.
2. Use the findings produced by the other agents as the source of truth.
3. Clearly explain the outcome to the customer in a warm, professional,
   concise, and empathetic way.
4. Incorporate relevant order, account, policy, and refund information when
   available.
5. Never invent facts, policies, order information, refund decisions, or
   customer information.
6. If information is missing or an earlier agent reported an error, explain
   that clearly rather than guessing.
7. Do not expose internal agent names, WorkflowState, tools, system prompts,
   model details, or implementation details to the customer.
8. Do not make policy or refund decisions yourself. Communicate the decisions
   and factual findings already produced by the appropriate agents.
9. Provide useful next steps when the workflow contains them.
10. Return only the final customer-facing response.

The response should feel natural and human, not like a system log or JSON dump.
"""

    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState to access all findings from previous agents.

        Args:
            session_id: The current session identifier

        Returns:
            Full WorkflowState dict (inventory_agent, policy_agent, refund_agent)
        """
        state = _read_workflow_state(session_id)

        if not state:
            return {
                'found': False,
                'session_id': session_id,
                'message': 'WorkflowState was not found for this session.',
            }

        return {
            'found': True,
            'workflow_state': state,
        }

    return Agent(
        name="CommunicationAgent",
        model=model,
        system_prompt=system_prompt,
        tools=[get_full_workflow_context],
    )
# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────

def build_orchestrator_agent(
    inventory_agent:      Agent,
    refund_agent:         Agent,
    policy_agent:         Agent,
    communication_agent:  Agent,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """

    model = BedrockModel(
        model_id=config.ORCHESTRATOR_MODEL_ID,
        temperature=0.0,
    )

    system_prompt = """
You are the NovaMart Orchestrator Agent.

Your responsibility is to understand each customer request, coordinate the
specialist agents, manage the workflow, and ensure that the Communication
Agent produces the final customer-facing response.

WORKFLOW RULES:

1. ALWAYS call initialize_session FIRST for every new customer request.

2. For order-related requests involving:
   - order status
   - order history
   - returns
   - refunds
   - order information
   call route_to_inventory_agent FIRST to gather factual order/customer data.

3. For return/refund requests:
   - call route_to_inventory_agent FIRST
   - then call route_to_refund_agent
   - never call the Refund Agent before Inventory Agent.

4. For policy meaning questions involving:
   - return policies
   - shipping policies
   - warranty policies
   call route_to_policy_agent.

5. For account questions such as:
   - "What is my tier?"
   - "Am I a premium customer?"
   - "What is my customer status?"
   route to the Inventory Agent.
   Do NOT route account-tier questions to the Policy Agent.

6. For simple mathematical calculations that do not require customer,
   order, or policy information, answer directly without routing to a
   specialist agent.

7. ALWAYS call route_to_communication_agent LAST for every customer request
   after all relevant specialist work has completed.

8. The Communication Agent is responsible for the final customer-facing
   response. Do not compose the final customer response yourself.

9. Do not invent customer information, order information, policy rules,
   refund decisions, or other facts.

10. Use the specialist agents only when their information is relevant to the
    customer's request.

11. Maintain the workflow through WorkflowState. Routing tools are
    responsible for reading the current state and storing their results.

12. Never expose internal agent names, WorkflowState, tools, model details,
    or implementation details to the customer.
"""

    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route an order-related request to the Inventory Agent to gather order facts.
        Call this FIRST for any request involving order status, history, or returns.

        Args:
            session_id:  The current session identifier (from the customer request)
            customer_id: The customer's unique identifier
            request:     The customer's original request

        Returns:
            Inventory facts retrieved by the InventoryAgent
        """
        state = _read_workflow_state(session_id)

        if not state:
            raise RuntimeError(
                f"WorkflowState not found for session {session_id}. "
                "initialize_session must be called first."
            )

        expected_version = int(state["version"])

        trace.step_start("inventory_agent")

        result = inventory_agent(
            f"""
Customer ID: {customer_id}
Session ID: {session_id}
Customer request: {request}

Gather the factual customer, order, and order-history information relevant
to this request. Do not make refund or policy eligibility decisions.
"""
        )

        result_text = str(result)

        _update_workflow_state(
            session_id,
            {"inventory_agent": result_text},
            expected_version,
        )

        trace.step_done("inventory_agent", expected_version)

        return result_text

    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """
        Route a policy question to the Policy Agent (multi-agent RAG).
        Call this for questions about return policies, shipping, or warranties.

        Args:
            session_id:  The current session identifier
            request:    The customer's policy question

        Returns:
            Policy information retrieved and synthesized by PolicyAgent
        """
        state = _read_workflow_state(session_id)

        if not state:
            raise RuntimeError(
                f"WorkflowState not found for session {session_id}. "
                "initialize_session must be called first."
            )

        expected_version = int(state["version"])

        trace.step_start("policy_agent")

        result = policy_agent(
            f"""
Session ID: {session_id}
Customer policy question: {request}

Retrieve and synthesize the relevant NovaMart policy information.
Ground the answer in the retrieved policy documents.
"""
        )

        result_text = str(result)

        _update_workflow_state(
            session_id,
            {"policy_agent": result_text},
            expected_version,
        )

        trace.step_done("policy_agent", expected_version)

        return result_text

    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route a return/refund request to the Refund Agent.
        Call this AFTER route_to_inventory_agent has gathered order facts.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier
            request:     The return/refund request

        Returns:
            Refund decision from the RefundAgent
        """
        state = _read_workflow_state(session_id)

        if not state:
            raise RuntimeError(
                f"WorkflowState not found for session {session_id}. "
                "initialize_session must be called first."
            )

        inventory_context = state.get("inventory_agent")

        if not inventory_context:
            return (
                "Refund processing cannot continue because inventory/order "
                "information has not been gathered yet. Call "
                "route_to_inventory_agent first."
            )

        expected_version = int(state["version"])

        trace.step_start("refund_agent")

        result = refund_agent(
            f"""
Customer ID: {customer_id}
Session ID: {session_id}
Customer request: {request}

Use get_inventory_context to inspect the inventory findings already gathered
for this session. Determine return/refund eligibility according to the
applicable NovaMart rules and process the request when appropriate.
"""
        )

        result_text = str(result)

        _update_workflow_state(
            session_id,
            {"refund_agent": result_text},
            expected_version,
        )

        trace.step_done("refund_agent", expected_version)

        return result_text

    @tool
    def route_to_communication_agent(
        session_id: str,
        customer_id: str,
        original_request: str,
    ) -> str:
        """
        Route to the Communication Agent to compose the final customer response.
        Call this LAST - after all relevant worker agents have run.

        Args:
            session_id:       The current session identifier
            customer_id:      The customer's unique identifier
            original_request: The customer's original message

        Returns:
            Final customer-facing response drafted by CommunicationAgent
        """
        state = _read_workflow_state(session_id)

        if not state:
            raise RuntimeError(
                f"WorkflowState not found for session {session_id}. "
                "initialize_session must be called first."
            )

        expected_version = int(state["version"])

        trace.step_start("communication_agent")

        result = communication_agent(
            f"""
Customer ID: {customer_id}
Session ID: {session_id}
Original customer request: {original_request}

Read the complete WorkflowState using get_full_workflow_context and compose
the final customer-facing response.
"""
        )

        result_text = str(result)

        _update_workflow_state(
            session_id,
            {"communication_agent": result_text},
            expected_version,
        )

        trace.step_done("communication_agent", expected_version)

        return result_text

    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """
        Create a blank WorkflowState record at the start of each new session.
        Call this at the VERY BEGINNING of processing every customer request.

        Args:
            session_id:  A unique identifier for this session
            customer_id: The customer's identifier

        Returns:
            Confirmation that the session was initialized
        """
        existing_state = _read_workflow_state(session_id)

        if existing_state:
            return (
                f"Session {session_id} is already initialized "
                f"at workflow version {existing_state['version']}."
            )

        state = _create_workflow_state(
            session_id=session_id,
            customer_id=customer_id,
        )

        return (
            f"Session {session_id} initialized successfully "
            f"for customer {customer_id}. "
            f"Workflow version: {state['version']}."
        )

    return Agent(
        name="NovaMart-Orchestrator",
        model=model,
        system_prompt=system_prompt,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent,
        ],
    )
# ═══════════════════════════════════════════════════════
#  AGENT GRAPH HELPERS
# ═══════════════════════════════════════════════════════

def _apply_guardrail(agents: list) -> None:
    """
    Attach the Bedrock Guardrail (Task 3) to every agent's BedrockModel.
    Guardrails are enforced per model invocation, so once GUARDRAIL_ID /
    GUARDRAIL_VERSION are known (in .env locally, as runtime environment
    variables when deployed) every agent in the graph runs behind the
    guardrail - no change to the agents themselves is needed.
    """
    guardrail_id      = config.GUARDRAIL_ID
    guardrail_version = config.GUARDRAIL_VERSION
    if not guardrail_id or not guardrail_version:
        return
    for agent in agents:
        model = getattr(agent, 'model', None)
        if model is not None and hasattr(model, 'update_config'):
            model.update_config(guardrail_id=guardrail_id,
                                guardrail_version=guardrail_version)


def build_agent_graph(verbose: bool = False) -> Agent:
    """Build all five agents, apply the guardrail, return the orchestrator."""
    def _ok(label):
        if verbose:
            print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  {label}{_C.RESET}", flush=True)

    inventory_agent     = build_inventory_agent();     _ok('InventoryAgent')
    refund_agent        = build_refund_agent();        _ok('RefundAgent')
    policy_agent        = build_policy_agent();        _ok('PolicyAgent')
    communication_agent = build_communication_agent(); _ok('CommunicationAgent')
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    _ok('Orchestrator')
    _apply_guardrail([inventory_agent, refund_agent, policy_agent,
                      communication_agent, orchestrator])
    if verbose and config.GUARDRAIL_ID:
        print(f"  {_C.GRY}          Guardrail {config.GUARDRAIL_ID} "
              f"(v{config.GUARDRAIL_VERSION}) attached to all agents{_C.RESET}")
    return orchestrator


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT PACKAGING
#
#  AgentCore Runtime "direct code deployment" runs a zip that contains the
#  code AND every dependency, compiled for linux/arm64 and the Python version
#  selected in codeConfiguration.runtime - the runtime installs nothing.
#  build_deployment_package() downloads matching wheels with pip
#  (--platform/--python-version/--only-binary) and zips them together with
#  this file, config.py and the other src/ modules. Inside the runtime this
#  same file is the entry point: with no command-line argument it starts the
#  HTTP server (see run_serve) instead of printing usage.
# ═══════════════════════════════════════════════════════

RUNTIME_ENTRYPOINT   = 'agent_orchestrator.py'      # codeConfiguration.entryPoint
RUNTIME_PYTHON       = 'PYTHON_3_12'                # codeConfiguration.runtime
_RUNTIME_PY_VERSION  = '3.12'                       # must match RUNTIME_PYTHON
_RUNTIME_PLATFORM    = 'manylinux2014_aarch64'      # AgentCore runs on arm64
_RUNTIME_MARKER      = '.agentcore-runtime'         # tells __main__ to serve
_RUNTIME_REQUIREMENTS = ['strands-agents>=1.0', 'bedrock-agentcore>=0.1',
                         'boto3>=1.42', 'python-dotenv>=1.0']

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
_RUNTIME_PROJECT_FILES = [
    os.path.join(_SRC_DIR, 'agent_orchestrator.py'),
    os.path.join(_SRC_DIR, 'agent_utils.py'),
    os.path.join(_SRC_DIR, 'agent_observability.py'),
    os.path.join(_SRC_DIR, 'bedrock_kb_retrieval.py'),
    os.path.join(_ROOT_DIR, 'config.py'),
]


def _zip_write(zf, full: str, arcname: str, data: bytes = None) -> None:
    """Add one file with the 644/755 permissions AgentCore requires."""
    import zipfile
    info = zipfile.ZipInfo.from_file(full, arcname) if data is None else zipfile.ZipInfo(arcname)
    info.compress_type = zipfile.ZIP_DEFLATED
    executable = data is None and os.access(full, os.X_OK) and not full.endswith('.py')
    info.external_attr = ((0o755 if executable else 0o644) & 0xFFFF) << 16
    if data is None:
        with open(full, 'rb') as fh:
            data = fh.read()
    zf.writestr(info, data)


def build_deployment_package(output_path: str) -> str:
    """Build the AgentCore deployment zip at output_path and return the path."""
    import shutil, subprocess, tempfile, zipfile

    missing = [p for p in _RUNTIME_PROJECT_FILES if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Cannot package runtime, missing: {missing}")

    with tempfile.TemporaryDirectory(prefix='agentcore-pkg-') as tmp:
        deps_dir = os.path.join(tmp, 'deps')
        os.makedirs(deps_dir)
        print(f"  Downloading arm64 dependencies (python {_RUNTIME_PY_VERSION}) ...", flush=True)
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
            '--target', deps_dir, '--platform', _RUNTIME_PLATFORM,
            '--python-version', _RUNTIME_PY_VERSION, '--implementation', 'cp',
            '--only-binary=:all:', '--upgrade', *_RUNTIME_REQUIREMENTS,
        ], check=True)
        for junk in ('bin', 'tests'):
            shutil.rmtree(os.path.join(deps_dir, junk), ignore_errors=True)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for dirpath, dirnames, filenames in os.walk(deps_dir):
                dirnames[:] = [d for d in dirnames if d != '__pycache__']
                for name in filenames:
                    if not name.endswith(('.pyc', '.pyo')):
                        full = os.path.join(dirpath, name)
                        _zip_write(zf, full, os.path.relpath(full, deps_dir))
            for path in _RUNTIME_PROJECT_FILES:
                _zip_write(zf, path, os.path.basename(path))
            _zip_write(zf, '', _RUNTIME_MARKER, data=b'agentcore runtime package\n')

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Package built: {output_path} ({size_mb:.1f} MB, entry point {RUNTIME_ENTRYPOINT})")
    if size_mb > 250:
        raise RuntimeError("Deployment package exceeds the 250 MB AgentCore limit")
    return output_path


# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════

def create_guardrail() -> tuple[str, str]:
    """
    Create a Bedrock Guardrail for enterprise safety enforcement.

    Blocks harmful content, PII exposure, off-topic subjects, and profanity.
    Returns (guardrail_id, guardrail_version).
    """
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)

    # Reuse an existing guardrail when its ID and version are already
    # configured. This avoids ListGuardrails/GetGuardrail permission issues
    # in restricted AWS VocLabs environments.
    if config.GUARDRAIL_ID and config.GUARDRAIL_VERSION:
        print(
            f"Using configured Guardrail: "
            f"{config.GUARDRAIL_ID} (version: {config.GUARDRAIL_VERSION})"
        )
        return config.GUARDRAIL_ID, config.GUARDRAIL_VERSION


    # TODO: Create the guardrail
    # Use bedrock_client.create_guardrail() with:
    #   - name (config.GUARDRAIL_NAME) and description
    #   - contentPolicyConfig - filtersConfig for SEXUAL, VIOLENCE, HATE at HIGH
    #     strength and INSULTS, MISCONDUCT at MEDIUM strength (input + output)
    #   - sensitiveInformationPolicyConfig - piiEntitiesConfig:
    #       CREDIT_DEBIT_CARD_NUMBER and US_SOCIAL_SECURITY_NUMBER -> BLOCK
    #       EMAIL and PHONE -> ANONYMIZE
    #   - topicPolicyConfig - one DENY topic per entry in config.GUARDRAIL_BLOCKED_TOPICS
    #     (competitor products, pricing negotiations, legal threats)
    #   - wordPolicyConfig - managedWordListsConfig with type PROFANITY
    #   - blockedInputMessaging and blockedOutputsMessaging
    #
    # Then promote it from DRAFT to a numbered version with
    # bedrock_client.create_guardrail_version(guardrailIdentifier=...)
    # and return (guardrail_id, guardrail_version).

    response = bedrock_client.create_guardrail(
        name=config.GUARDRAIL_NAME,
        description=(
            "NovaMart enterprise customer support guardrail. "
            "Blocks harmful content, sensitive information exposure, "
            "off-topic requests, and profanity."
        ),
        contentPolicyConfig={
            'filtersConfig': [
                {
                    'type': 'SEXUAL',
                    'inputStrength': 'HIGH',
                    'outputStrength': 'HIGH',
                },
                {
                    'type': 'VIOLENCE',
                    'inputStrength': 'HIGH',
                    'outputStrength': 'HIGH',
                },
                {
                    'type': 'HATE',
                    'inputStrength': 'HIGH',
                    'outputStrength': 'HIGH',
                },
                {
                    'type': 'INSULTS',
                    'inputStrength': 'MEDIUM',
                    'outputStrength': 'MEDIUM',
                },
                {
                    'type': 'MISCONDUCT',
                    'inputStrength': 'MEDIUM',
                    'outputStrength': 'MEDIUM',
                },
            ]
        },
        sensitiveInformationPolicyConfig={
            'piiEntitiesConfig': [
                {
                    'type': 'CREDIT_DEBIT_CARD_NUMBER',
                    'action': 'BLOCK',
                },
                {
                    'type': 'US_SOCIAL_SECURITY_NUMBER',
                    'action': 'BLOCK',
                },
                {
                    'type': 'EMAIL',
                    'action': 'ANONYMIZE',
                },
                {
                    'type': 'PHONE',
                    'action': 'ANONYMIZE',
                },
            ]
        },
        topicPolicyConfig={
            'topicsConfig': [
                {
                    'name': topic,
                    'definition': (
                        f"Requests or discussions concerning {topic} "
                        "that are outside NovaMart customer support."
                    ),
                    'examples': [
                        f"Tell me about {topic}.",
                        f"Help me with {topic}.",
                    ],
                    'type': 'DENY',
                }
                for topic in config.GUARDRAIL_BLOCKED_TOPICS
            ]
        },
        wordPolicyConfig={
            'managedWordListsConfig': [
                {
                    'type': 'PROFANITY',
                }
            ]
        },
        blockedInputMessaging=(
            "I'm sorry, but I can't help with that request. "
            "Please ask a NovaMart customer-support question."
        ),
        blockedOutputsMessaging=(
            "I'm sorry, but I can't provide that response. "
            "Please ask a NovaMart customer-support question."
        ),
    )

    guardrail_id = response['guardrailId']

    version_response = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description="Initial production guardrail version",
    )

    guardrail_version = version_response['version']

    print(
        f"Guardrail created: {guardrail_id} "
        f"(version: {guardrail_version})"
    )

    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """
    Deploy the multi-agent system to Amazon Bedrock AgentCore Runtime.

    AgentCore does not serialize Python objects, so `orchestrator_agent` is
    not uploaded directly. Instead the packaging step below zips this file,
    which doubles as the HTTP entry point (see run_serve), together with its
    helper modules and all dependencies
    compiled for arm64. The runtime is then created from that zip ("direct
    code deployment").

    The guardrail is attached by environment variables: inside the runtime
    build_agent_graph() reads GUARDRAIL_ID / GUARDRAIL_VERSION and applies
    them to every agent's model (see _apply_guardrail), exactly as `test`
    and `chat` do locally.

    Returns:
        The AgentCore Runtime ARN
    """
    runtime_name = config.AGENTCORE_RUNTIME_NAME
    s3_client    = boto3.client('s3', region_name=config.AWS_REGION)

    # Check if runtime already exists
    try:
        existing = agentcore_control.list_agent_runtimes()
        for r in existing.get('agentRuntimes', []):
            if r['agentRuntimeName'] == runtime_name:
                runtime_arn = r['agentRuntimeArn']
                print(f"AgentCore Runtime already exists: {runtime_arn}")
                return runtime_arn
    except Exception as e:
        print(f"  [Note] Could not check existing runtimes: {e}")

    print(f"  AWS Account: {config.ACCOUNT_ID}  |  Region: {config.AWS_REGION}")

    # Build the deployment package and upload it to S3.
    package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'build', 'deployment_package.zip')
    build_deployment_package(package_path)

    artifact_key = f"agentcore-artifacts/{runtime_name}/deployment_package.zip"
    s3_client.upload_file(package_path, config.POLICY_BUCKET, artifact_key)
    print(f"  Artifact uploaded: s3://{config.POLICY_BUCKET}/{artifact_key}")

        # Deploy to AgentCore Runtime using direct S3 code deployment.
    response = agentcore_control.create_agent_runtime(
        agentRuntimeName=runtime_name,
        description=(
            "NovaMart multi-agent e-commerce customer support system "
            "deployed with Amazon Bedrock AgentCore Runtime."
        ),
        roleArn=config.AGENTCORE_ROLE_ARN,
        agentRuntimeArtifact={
            'codeConfiguration': {
                'code': {
                    's3': {
                        'bucket': config.POLICY_BUCKET,
                        'prefix': artifact_key,
                    }
                },
                'runtime': RUNTIME_PYTHON,
                'entryPoint': [RUNTIME_ENTRYPOINT],
            }
        },
        networkConfiguration={
            'networkMode': 'PUBLIC'
        },
        protocolConfiguration={
            'serverProtocol': 'HTTP'
        },
        environmentVariables={
            'AWS_REGION': config.AWS_REGION,
            'PROJECT_NAME': config.PROJECT_NAME,
            'RETURNS_KB_ID': config.RETURNS_KB_ID,
            'SHIPPING_KB_ID': config.SHIPPING_KB_ID,
            'WARRANTY_KB_ID': config.WARRANTY_KB_ID,
            'AGENT_LOG_GROUP': config.AGENT_LOG_GROUP,
            'GUARDRAIL_ID': guardrail_id,
            'GUARDRAIL_VERSION': guardrail_version,
        },
    )

    if response is None:
        raise NotImplementedError("deploy_to_agentcore_runtime: create_agent_runtime() not implemented")

    # Wait for the runtime to become READY and return its ARN.
    runtime_arn = response['agentRuntimeArn']
    print(f"  Runtime created: {runtime_arn}")
    print("  Waiting for runtime status READY", end='', flush=True)
    wait_for_runtime_ready(agentcore_control, response['agentRuntimeId'])
    print(' ready.')
    return runtime_arn


# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Create an AgentCore Memory resource for session-scoped conversational
    context. Uses the SESSION_SUMMARY (summaryMemoryStrategy) strategy with
    7-day event retention.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAME
    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m['id'].startswith(memory_name):
            memory_arn = m['arn']
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn

    # TODO: Create AgentCore Memory
    # Use agentcore_control.create_memory() with:
    #   - name (memory_name) and a description
    #   - eventExpiryDuration = 7   (days)
    #   - memoryStrategies = [{'summaryMemoryStrategy': {
    #         'name': 'SessionSummary',
    #         'namespaces': ['/summaries/{actorId}/{sessionId}']}}]
    #   - clientToken (e.g. str(uuid.uuid4())) for idempotency
    # Store the API response in `response`.
    response = agentcore_control.create_memory(
    name=config.MEMORY_NAME,
    description=(
        "NovaMart session-scoped conversational memory "
        "for multi-agent customer support."
    ),
    eventExpiryDuration=7,
    memoryStrategies=[
        {
            "summaryMemoryStrategy": {
                "name": "SessionSummary",
                "namespaces": [
                    "/summaries/{actorId}/{sessionId}"
                ],
            }
        }
    ],
    clientToken=str(uuid.uuid4()),
)

    if response is None:
        raise NotImplementedError("configure_memory: create_memory() not implemented")

    # Wait until the memory resource is ACTIVE and return its ARN.
    memory = response['memory']
    print(f"  Memory created: {memory['arn']}  (status: {memory['status']})")
    print("  Waiting for memory status ACTIVE", end='', flush=True)
    deadline = time.time() + 300
    while memory['status'] != 'ACTIVE' and time.time() < deadline:
        time.sleep(10)
        print('.', end='', flush=True)
        memory = agentcore_control.get_memory(memoryId=memory['id'])['memory']
        if memory['status'] == 'FAILED':
            raise RuntimeError(f"Memory creation failed: {memory.get('failureReason')}")
    print(' ready.' if memory['status'] == 'ACTIVE' else f" status {memory['status']}")
    return memory['arn']


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════

def configure_observability(runtime_arn: str) -> None:
    """
    Configure observability for the deployed agent:
    - Agent logs → CloudWatch Logs at INFO level (config.AGENT_LOG_GROUP)
    - Execution traces → AWS X-Ray at 100% sampling

    The loggingConfiguration built here is applied by
    apply_observability_config() (agent_observability.py):
      cloudWatchConfig -> log group created; runtime env AGENT_LOG_GROUP /
                          AGENT_LOG_LEVEL so the deployed agent ships its logs there
      xRayConfig       -> CloudWatch Transaction Search enabled with the given
                          sampling percentage; runtime env AGENT_TRACING_ENABLED /
                          AGENT_TRACE_SAMPLING_RATE
    """

    """
    Configure CloudWatch logging and AWS X-Ray tracing for the runtime.
    """
    logging_configuration = {
        "cloudWatchConfig": {
            "logGroupName": config.AGENT_LOG_GROUP,
            "logLevel": "INFO",
            "enabled": True,
        },
        "xRayConfig": {
            "enabled": True,
            "samplingRate": 1.0,
        },
    }

    try:
        summary = apply_observability_config(
            runtime_arn,
            logging_configuration,
        )

        print(
            f"  CloudWatch logging: {config.AGENT_LOG_GROUP} "
            f"(INFO)"
        )
        print("  X-Ray tracing: enabled (100% sampling)")

    except Exception as e:
     print(f"[Note] Observability configuration failed: {e}")


# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT
#
#  Production equivalent of in-process @tool functions.
#  Registers Lambda-backed tools on a managed MCP endpoint so tools
#  can be independently deployed, versioned, and discovered at runtime.
#
#  Deployment pattern:
#    Local dev  → LambdaGateway + gateway.register_target(...)
#    Production → deploy_agentcore_gateway() using real AWS API
#
#  Requires Lambda tool functions to be deployed separately.
#  Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env
#  to the deployed Lambda function names.
# ═══════════════════════════════════════════════════════

# Lambda function names for gateway tool backends (set in .env after deploying)
_ORDERS_FUNCTION    = os.environ.get('ORDERS_FUNCTION',    f"{config.PROJECT_NAME}-orders-api")
_POLICY_FUNCTION    = os.environ.get('POLICY_FUNCTION',    f"{config.PROJECT_NAME}-policy-api")
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', f"{config.PROJECT_NAME}-customers-api")


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Production equivalent of the in-process @tool functions defined inside
    build_*_agent(). Each tool becomes a Lambda function registered as a
    gateway target; agents discover tools at runtime via the MCP endpoint —
    no code changes needed when adding or updating tools.

    Uses a three-step deployment pattern:
      1. create_gateway  (MCP protocol, SEMANTIC search)
      2. create_gateway_target  (one per Lambda-backed tool)
      3. Agents connect via the returned gateway_url

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        dict with gateway_id, gateway_url, and status.
    """
    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    print(f"\n  Registering {len(targets)} Gateway targets...")
    for t in targets:
        try:
            lambda_arn = _gw_get_function_arn(t['function'])
            _gw_create_target(agentcore_ctrl, gateway_id, t, lambda_arn)
        except Exception as e:
            print(f"    [Skipped] {t['name']}: {e}")

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url, 'status': 'CREATING'}



# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> dict:
    """
    Invoke the deployed agent via AgentCore Runtime (see run_serve).

    AgentCore requires runtimeSessionId to be at least 33 characters, so the
    short project session id is embedded in a longer, unique runtime session id.
    """
    if not config.AGENTCORE_RUNTIME_ARN:
        raise RuntimeError("AGENTCORE_RUNTIME_ARN is not set - run the deploy command first")

    runtime_session_id = f"{session_id}-{uuid.uuid4().hex}"     # >= 33 chars
    payload = json.dumps({
        'prompt':      user_message,
        'session_id':  session_id,
        'customer_id': customer_id,
    })
    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        runtimeSessionId=runtime_session_id,
        contentType='application/json',
        accept='application/json',
        payload=payload,
    )
    body = response['response'].read()
    try:
        return json.loads(body)
    except (TypeError, ValueError):
        return {'result': body.decode('utf-8', errors='replace') if isinstance(body, bytes) else str(body)}


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        print(f"  Gateway URL : {gw['gateway_url']}")
        print(f"  Agents connect via MCP at this endpoint — no code changes needed")
    except Exception as e:
        print(f"  [Note] Gateway deployment skipped: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    print(f"  Then try the deployed runtime:")
    print(f"  python src/agent_orchestrator.py invoke \"What is the return policy for premium customers?\"\n")
    return runtime_arn, guardrail_id


# ═══════════════════════════════════════════════════════
#  LOCAL TEST SCENARIOS
# ═══════════════════════════════════════════════════════

# Order IDs match infrastructure/seed_data.py.
TEST_CASES = [
    ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
    ("CUST-002", "What is the return policy for premium customers?"),
    ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
]

# Test customers shown by the chat command. Data matches seed_data.py.
TEST_CUSTOMERS = [
    ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Wireless Headphones Pro"),
    ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "Mechanical Keyboard K2"),
    ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "Laptop UltraBook 14"),
    ("CUST-004", "David Lee",     "Standard", "ORD-30001", "Phone Case Slim"),
]


def run_test_scenarios() -> None:
    """Run the three scenarios locally; every request is traced to X-Ray."""
    print("Running local agent test...")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph()

    for customer_id, query in TEST_CASES:
        session_id = str(uuid.uuid4())[:8]
        print(f"\n{'─'*60}")
        print(f"Session: {session_id} | Customer: {customer_id}")
        print(f"Query: {query}")
        prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
        with tracer.trace_request(session_id, customer_id, query):
            response = orchestrator(prompt)
        print(f"Response: {response}")
        print_trace_hint()
    flush_logs()


def run_chat() -> None:
    """Interactive terminal chat - educational mode."""
    W = _C.W

    # ── Welcome banner ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
    print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

    # ── Test customers ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.BOLD}Test Customers{_C.RESET}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
    print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
    for cid, name, tier, order, product in TEST_CUSTOMERS:
        tier_col = _C.INV if tier == 'Premium' else _C.GRY
        print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
              f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print()

    customer_id = (
        input(f"  Enter Customer ID (default: CUST-001): ").strip()
        or "CUST-001"
    )
    session_id  = str(uuid.uuid4())[:8]
    print()
    print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
    print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
    print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
    print()

    # ── Build agents and show initialization order.
    print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph(verbose=True)
    print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
    print()

    # ── Conversation loop ─────────────────────────────────────────────
    while True:
        try:
            user_input = input(
                f"  {_C.BOLD}You >{_C.RESET} "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print(f"  {_C.GRY}Session ended.{_C.RESET}")
            break

        prompt  = (f"[Session ID: {session_id}] "
                   f"[Customer ID: {customer_id}] {user_input}")
        t0_turn = time.time()

        # ── Install proxy, run orchestrator (traced), restore stdout ───
        trace.new_turn()
        sys.stdout = _trace_writer
        try:
            with tracer.trace_request(session_id, customer_id, user_input):
                response = orchestrator(prompt)
        finally:
            sys.stdout = _real_stdout   # always restore, even on exception

        elapsed = time.time() - t0_turn

        # ── Resolve the final customer-facing text ────────────────────
        final_state = _read_workflow_state(session_id) or {}
        comm_result = final_state.get('communication_agent', '')
        text = _strip_xml_tags(comm_result or str(response))

        # ── DynamoDB workflow state summary ───────────────────────────
        trace.summary(session_id, elapsed)

        # ── Final customer-facing response ────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        for line in text.splitlines():
            print(f"  {line}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        if tracer.last_trace_id:
            print(f"  {_C.GRY}X-Ray trace : {tracer.last_trace_id}"
                  f"{'' if tracer.last_published else '  (not published)'}{_C.RESET}")
        print()
    flush_logs()


def run_invoke(message: str, customer_id: str = "CUST-001") -> None:
    """Send one message to the deployed AgentCore Runtime and print the reply."""
    session_id = str(uuid.uuid4())[:8]
    print(f"Invoking {config.AGENTCORE_RUNTIME_ARN}")
    print(f"Session: {session_id} | Customer: {customer_id}")
    print(f"Query: {message}\n")
    result = invoke_agent(session_id, customer_id, message)
    print(f"Response: {result.get('result', result)}")
    if result.get('trace_id'):
        print(f"X-Ray trace: {result['trace_id']}")


def run_serve() -> None:
    """
    HTTP entry point executed inside Amazon Bedrock AgentCore Runtime.

    BedrockAgentCoreApp (bedrock-agentcore SDK) exposes the contract the
    runtime expects - POST /invocations and GET /ping on port 8080 - and hands
    each request payload to the function decorated with @app.entrypoint.

    Request payload (see invoke_agent):
        {"prompt": "<customer message>", "customer_id": "CUST-001", "session_id": "abc12345"}
    Response:
        {"result": "<final customer-facing text>", "session_id": ..., "trace_id": ...}

    The five-agent graph is built once (first request) and reused. Guardrail,
    tracing and logging are applied exactly as in the local test/chat modes,
    from the runtime's environment variables.
    """
    from bedrock_agentcore import BedrockAgentCoreApp

    os.environ.setdefault('AGENT_RUNTIME_MODE', 'agentcore-runtime')
    if os.environ.get('AGENT_LOG_GROUP') and 'AGENT_LOG_TO_CLOUDWATCH' not in os.environ:
        os.environ['AGENT_LOG_TO_CLOUDWATCH'] = 'true'

    app   = BedrockAgentCoreApp()
    lock  = threading.Lock()
    graph = {}

    def _orchestrator():
        with lock:
            if 'agent' not in graph:
                setup_logging()
                graph['agent'] = build_agent_graph()
        return graph['agent']

    @app.entrypoint
    def invoke(payload, context=None):
        payload     = payload or {}
        prompt      = payload.get('prompt') or payload.get('message') or ''
        customer_id = payload.get('customer_id') or 'CUST-001'
        session_id  = payload.get('session_id') or (
            getattr(context, 'session_id', None) or uuid.uuid4().hex)[:8]
        if not prompt:
            return {'error': "payload must include 'prompt'"}

        enriched = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {prompt}"
        with tracer.trace_request(session_id, customer_id, prompt):
            response = _orchestrator()(enriched)

        state = _read_workflow_state(session_id) or {}
        text  = _strip_xml_tags(state.get('communication_agent', '') or str(response))
        flush_logs()
        return {'result': text, 'session_id': session_id, 'customer_id': customer_id,
                'trace_id': tracer.last_trace_id}

    app.run()


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''

    # Inside the AgentCore Runtime package (marker file next to this script)
    # the entry point is started without arguments -> serve HTTP.
    if not command and os.path.exists(os.path.join(_SRC_DIR, _RUNTIME_MARKER)):
        command = 'serve'

    if command == 'deploy':
        deploy_all()

    elif command == 'serve':
        run_serve()

    elif command == 'test':
        run_test_scenarios()

    elif command == 'chat':
        run_chat()

    elif command == 'invoke':
        if len(sys.argv) < 3:
            print('Usage: python src/agent_orchestrator.py invoke "<message>" [CUSTOMER_ID]')
            sys.exit(1)
        run_invoke(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "CUST-001")

    else:
        print("Usage:")
        print("  python src/agent_orchestrator.py deploy           # Deploy to AgentCore (Tasks 3-6)")
        print("  python src/agent_orchestrator.py test             # Run the 3 test scenarios locally")
        print("  python src/agent_orchestrator.py chat             # Interactive terminal chat")
        print("  python src/agent_orchestrator.py invoke \"<msg>\"   # Call the deployed runtime")
        print("  python src/agent_orchestrator.py serve            # HTTP server (used inside AgentCore Runtime)")
