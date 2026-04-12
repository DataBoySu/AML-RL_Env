# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Aml Env Environment.

The AML_env environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Literal, Optional, Any, Union

# ==========================================
# OBSERVATION SPACE
# ==========================================
class AmlObservation(Observation):
    alert_details: str = Field(description="The constant mission objective and initial alert.")
    budget_remaining: int = Field(description="API calls remaining.")
    last_action: Optional[str] = Field(default=None, description="Last tool used.")
    last_action_result: Optional[Any] = Field(default=None, description="Payload returned by the API.")
    error_message: Optional[str] = Field(default=None, description="Error string if action failed.")

# ==========================================
# ACTION SPACE
# ==========================================
class QueryTransactions(Action):
    action_type: Literal["query_transactions"]
    account_id: str = Field(description="The exact ACC-XXXX ID to query.")
    limit: int = Field(default=10, description="Max transactions to return.")
    offset: int = Field(default=0, description="Offset for pagination.")

class SearchTransactions(Action):
    action_type: Literal["search_transactions"]
    account_id: str = Field(description="The exact ACC-XXXX ID to query.")
    keyword: str = Field(description="Keyword to search in memo_text.")

class GetKYCRecord(Action):
    action_type: Literal["get_kyc_record"]
    entity_id: str = Field(description="The exact ENT-XXXX ID to look up.")

class SubmitDecision(Action):
    action_type: Literal["submit_decision"]
    decision: Literal["FRAUD", "CLEAR"] = Field(description="Your final verdict.")
    evidence_links: List[str] = Field(description="List of ACC-XXXX or ENT-XXXX IDs proving fraud.")

# The master Action model using Union
class AmlAction(Action):
    action: Union[QueryTransactions, SearchTransactions, GetKYCRecord, SubmitDecision] = Field(
        discriminator='action_type'
    )
    
