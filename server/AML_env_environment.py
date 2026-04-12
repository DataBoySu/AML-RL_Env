# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AML Investigator Environment Implementation.

An OpenEnv-compatible mock financial system that forces the agent to
explore a massive transaction graph using a strict budget.
"""

import json
import os
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        AmlAction, 
        AmlObservation,
        QueryTransactions,
        SearchTransactions,
        GetKYCRecord,
        SubmitDecision
    )
except ImportError:
    from models import (
        AmlAction, 
        AmlObservation,
        QueryTransactions,
        SearchTransactions,
        GetKYCRecord,
        SubmitDecision
    )

# The strict API call limit. Forces the agent to be efficient.
MAX_BUDGET = 15

class AmlEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the AML environment and load the mock database."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.budget_remaining = MAX_BUDGET
        self.alert_details = ""
        
        # Determine the data directory robustly
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data"
        
        # Load the JSON files into fast O(1) lookup dictionaries
        try:
            with open(data_dir / "entities.json", "r") as f:
                ent_data = json.load(f)
                ent_list = ent_data.get("entities", ent_data) if isinstance(ent_data, dict) else ent_data
                self.entities_db = {e["entity_id"]: e for e in ent_list}
                
            with open(data_dir / "accounts.json", "r") as f:
                acc_data = json.load(f)
                acc_list = acc_data.get("accounts", acc_data) if isinstance(acc_data, dict) else acc_data
                self.accounts_db = {a["account_id"]: a for a in acc_list}
                
            with open(data_dir / "transactions.json", "r") as f:
                txn_data = json.load(f)
                txn_list = txn_data.get("transactions", txn_data) if isinstance(txn_data, dict) else txn_data
                # Sort transactions by timestamp to ensure deterministic pagination
                self.transactions_db = sorted(txn_list, key=lambda x: x.get("timestamp", ""))
                
            print(f"[AML-ENV] Loaded {len(self.entities_db)} entities, {len(self.accounts_db)} accounts, {len(self.transactions_db)} transactions.")
        except Exception as e:
            print(f"[AML-ENV ERROR] Failed to load data from {data_dir}. Ensure JSON files exist. Error: {e}")
            self.entities_db = {}
            self.accounts_db = {}
            self.transactions_db = []

    def reset(self, task: str = "aml_easy", **kwargs) -> AmlObservation:
        """
        Reset the environment and set the specific alert based on the task_id.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        budgets = {
            "aml_easy": 5,   # Barely enough to make 2 queries and submit
            "aml_medium": 10, # Requires some pagination/exploration
            "aml_hard": 20    # Deep investigation required
        }
        self.budget_remaining = budgets.get(task, 15)

        # The mission strings match our yaml descriptions
        alerts = {
            "aml_easy": "System Alert: ACC-9001 flagged for high-risk transfer. Investigate and submit decision.",
            "aml_medium": "System Alert: ACC-9010 flagged for anomalous deposit velocity. Investigate for structuring.",
            "aml_hard": "System Alert: ACC-9021 flagged for large outbound transfer. Investigate for hidden circular loops."
        }
        self.alert_details = alerts.get(task, f"System Alert: Investigate task {task}.")

        return AmlObservation(
            alert_details=self.alert_details,
            budget_remaining=self.budget_remaining,
            last_action=None,
            last_action_result=None,
            error_message=None,
            done=False,
            reward=0.0
        )

    def step(self, action: AmlAction) -> AmlObservation: 
        """
        The reactive state machine. Intercepts the Pydantic action, queries 
        the JSON data, and handles errors as strings.
        """
        self._state.step_count += 1
        self.budget_remaining -= 1
        
        # Default step penalty to penalize random infinite looping
        reward = -0.02
        done = False
        result_data = None
        error_msg = None
        
        # Extract the specific tool from the Union wrapper
        tool = action.action
        tool_name = tool.action_type

        try:
            # ---------------------------------------------------------
            # TOOL 1: Query Transactions
            # ---------------------------------------------------------
            if isinstance(tool, QueryTransactions):
                acc_id = tool.account_id
                if acc_id not in self.accounts_db:
                    raise ValueError(f"Account '{acc_id}' not found in registry.")
                
                # Filter related transactions
                related_txns = [
                    t for t in self.transactions_db 
                    if t["sender_account"] == acc_id or t["receiver_account"] == acc_id
                ]
                
                # Apply pagination (Context Compaction)
                start = tool.offset
                end = start + tool.limit
                result_data = related_txns[start:end]
                
            # ---------------------------------------------------------
            # TOOL 2: Search Transactions
            # ---------------------------------------------------------
            elif isinstance(tool, SearchTransactions):
                acc_id = tool.account_id
                if acc_id not in self.accounts_db:
                    raise ValueError(f"Account '{acc_id}' not found in registry.")
                
                keyword = tool.keyword.lower()
                related_txns = [
                    t for t in self.transactions_db 
                    if (t["sender_account"] == acc_id or t["receiver_account"] == acc_id) 
                    and keyword in t.get("memo_text", "").lower()
                ]
                result_data = related_txns
                
            # ---------------------------------------------------------
            # TOOL 3: Get KYC Record
            # ---------------------------------------------------------
            elif isinstance(tool, GetKYCRecord):
                ent_id = tool.entity_id
                if ent_id not in self.entities_db:
                    raise ValueError(f"Entity '{ent_id}' not found in registry.")
                result_data = self.entities_db[ent_id]
                
            # ---------------------------------------------------------
            # TOOL 4: Submit Decision
            # ---------------------------------------------------------
            elif isinstance(tool, SubmitDecision):
                result_data = f"Decision '{tool.decision}' recorded with evidence: {tool.evidence_links}"
                done = True

        except Exception as e:
            # "Errors are data" - We catch bad inputs and feed them back to the agent
            error_msg = f"API Error: {str(e)}"
            result_data = None

        # Check for budget failure
        if self.budget_remaining <= 0 and not done:
            done = True
            error_msg = error_msg or "Investigation Budget Exhausted. Forced Termination."

        return AmlObservation(
            alert_details=self.alert_details,
            budget_remaining=self.budget_remaining,
            last_action=tool_name,
            last_action_result=result_data,
            error_message=error_msg,
            done=done,
            reward=reward
        )

    @property
    def state(self) -> State:
        return self._state