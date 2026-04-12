# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AML Investigator Environment Client.

High-level WebSocket client that wraps the OpenEnv EnvClient base class
with AML-specific action/observation types.
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AmlAction, AmlObservation


class AmlEnv(EnvClient[AmlAction, AmlObservation, State]):
    """
    WebSocket client for the AML Investigator environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step investigations with lower per-step latency.

    Example (Docker):
        >>> client = AmlEnv.from_docker_image("aml-env:latest")
        >>> try:
        ...     obs = client.reset(task="aml_easy")
        ...     result = client.step(AmlAction(action={
        ...         "action_type": "query_transactions",
        ...         "account_id": "ACC-9001"
        ...     }))
        ...     print(result.observation.last_action_result)
        ... finally:
        ...     client.close()

    Example (existing server):
        >>> with AmlEnv(base_url="http://localhost:7860") as env:
        ...     obs = env.reset(task="aml_easy")
        ...     result = env.step(AmlAction(action={
        ...         "action_type": "submit_decision",
        ...         "decision": "CLEAR",
        ...         "evidence_links": []
        ...     }))
    """

    def _step_payload(self, action: AmlAction) -> Dict:
        """
        Serialize AmlAction to the JSON dict sent over the WebSocket.

        Args:
            action: Typed AmlAction wrapper containing the specific tool call.

        Returns:
            Dict with the nested ``action`` key the server expects.
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[AmlObservation]:
        """
        Deserialize the server's JSON response into a typed StepResult.

        Args:
            payload: Raw JSON response dict from the server.

        Returns:
            StepResult containing an AmlObservation.
        """
        obs_data = payload.get("observation", {})
        observation = AmlObservation(
            alert_details=obs_data.get("alert_details", ""),
            budget_remaining=obs_data.get("budget_remaining", 0),
            last_action=obs_data.get("last_action"),
            last_action_result=obs_data.get("last_action_result"),
            error_message=obs_data.get("error_message"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Deserialize the server's /state response into a State object.

        Args:
            payload: Raw JSON response dict from the server.

        Returns:
            State with episode_id and step_count.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
