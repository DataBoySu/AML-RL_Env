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
from pydantic import ConfigDict, Field, field_validator
from typing import List, Literal, Optional, Any, Union

# ==========================================
# OBSERVATION SPACE
# ==========================================
class AmlObservation(Observation):
    model_config = ConfigDict(extra="forbid", strict=True)

    alert_details: str = Field(description="The constant mission objective and initial alert.")
    budget_remaining: int = Field(description="API calls remaining.")
    last_action: Optional[str] = Field(default=None, description="Last tool used.")
    last_action_result: Optional[Any] = Field(default=None, description="Payload returned by the API.")
    error_message: Optional[str] = Field(default=None, description="Error string if action failed.")

# ==========================================
# ACTION SPACE
# ==========================================
class QueryTransactions(Action):
    model_config = ConfigDict(extra="forbid", strict=True)

    action_type: Literal["query_transactions"]
    account_id: str = Field(pattern=r"^ACC-\d{4}$", description="The exact ACC-XXXX ID to query.")
    limit: int = Field(default=10, ge=1, le=100, description="Max transactions to return.")
    offset: int = Field(default=0, ge=0, description="Offset for pagination.")

class SearchTransactions(Action):
    model_config = ConfigDict(extra="forbid", strict=True)

    action_type: Literal["search_transactions"]
    account_id: str = Field(pattern=r"^ACC-\d{4}$", description="The exact ACC-XXXX ID to query.")
    keyword: str = Field(min_length=1, description="Keyword to search in memo_text.")

class GetKYCRecord(Action):
    model_config = ConfigDict(extra="forbid", strict=True)

    action_type: Literal["get_kyc_record"]
    entity_id: str = Field(pattern=r"^ENT-\d{4}$", description="The exact ENT-XXXX ID to look up.")

class SubmitDecision(Action):
    model_config = ConfigDict(extra="forbid", strict=True)

    action_type: Literal["submit_decision"]
    decision: Literal["FRAUD", "CLEAR"] = Field(description="Your final verdict.")
    evidence_links: List[str] = Field(
        default_factory=list,
        description="List of ACC-XXXX or ENT-XXXX IDs proving fraud.",
    )

# The master Action model using Union
class AmlAction(Action):
    model_config = ConfigDict(extra="forbid", strict=True)

    thought: str = Field(
        min_length=1,
        description="Short thinking pad with Observation: and Plan: sections.",
    )
    action: Union[QueryTransactions, SearchTransactions, GetKYCRecord, SubmitDecision] = Field(
        discriminator='action_type'
    )

    @field_validator("thought")
    @classmethod
    def thought_must_include_sections(cls, value: str) -> str:
        text = value.strip()
        lower_text = text.lower()
        if "observation:" not in lower_text or "plan:" not in lower_text:
            raise ValueError("thought must include 'Observation:' and 'Plan:' sections")
        return text
