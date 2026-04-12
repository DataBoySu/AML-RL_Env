---
title: AML Investigator — OpenEnv RL Environment
emoji: 🕵️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
tags:
  - openenv
---

# AML Investigator Environment

A financial crime investigation environment for Reinforcement Learning agents.
The agent must query a mock banking system (transactions, KYC records) under a strict API budget
to investigate flagged accounts and submit a final fraud/clear decision.

## Quick Start

The simplest way to use the Aml Env environment is through the `AmlEnv` class:

```python
from AML_env import AmlAction, AmlEnv

try:
    # Create environment from Docker image (built from root Dockerfile)
    env = AmlEnv.from_docker_image("aml-env:latest")

    # Reset to a specific task
    obs = env.reset(task="aml_easy")
    print(f"Alert: {obs.observation.alert_details}")
    print(f"Budget: {obs.observation.budget_remaining}")

    # Query transactions
    result = env.step(AmlAction(action={
        "action_type": "query_transactions",
        "account_id": "ACC-9001",
        "limit": 10,
        "offset": 0,
    }))
    print(f"Transactions: {result.observation.last_action_result}")

    # Submit final decision
    result = env.step(AmlAction(action={
        "action_type": "submit_decision",
        "decision": "CLEAR",
        "evidence_links": [],
    }))
    print(f"Done: {result.done}, Reward: {result.reward}")

finally:
    env.close()
```

That's it! The `AmlEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t aml-env:latest .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action Space
**AmlAction** wraps one of four tool calls (discriminated by `action_type`):

| Tool | Fields | Description |
|---|---|---|
| `query_transactions` | `account_id`, `limit`, `offset` | Paginated transaction history for an account |
| `search_transactions` | `account_id`, `keyword` | Search memo_text of transactions |
| `get_kyc_record` | `entity_id` | Retrieve KYC data for an entity |
| `submit_decision` | `decision` (`FRAUD`\|`CLEAR`), `evidence_links` | Final verdict — ends the episode |

### Observation Space
**AmlObservation** is returned after every `reset()` and `step()`:

| Field | Type | Description |
|---|---|---|
| `alert_details` | `str` | The investigation mission (constant per episode) |
| `budget_remaining` | `int` | API calls left before forced termination |
| `last_action` | `str \| None` | Name of the last tool called |
| `last_action_result` | `Any` | Payload returned by the last tool |
| `error_message` | `str \| None` | Error string if the last action failed |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Per-step reward signal |

### Reward
- **Per step:** `-0.02` (efficiency penalty discourages random looping)
- **Submit FRAUD (correct):** grader returns `0.4`–`1.0` depending on evidence quality
- **Submit CLEAR (correct false positive):** grader returns `1.0`
- **Budget exhausted without submission:** episode ends with accumulated negative rewards

## Advanced Usage

### Connecting to an Existing Server

If you already have a Aml Env environment server running, you can connect directly:

```python
from AML_env import AmlEnv

# Connect to existing server
AML_envenv = AmlEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = AML_envenv.reset()
result = AML_envenv.step(AmlAction(message="Hello!"))
```

Note: When connecting to an existing server, `AML_envenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from AML_env import AmlAction, AmlEnv

# Connect with context manager (auto-connects and closes)
with AmlEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(AmlAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    AmlEnvironment,  # Pass class, not instance
    AmlAction,
    AmlObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from AML_env import AmlAction, AmlEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with AmlEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(AmlAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/AML_env_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
AML_env/
├── Dockerfile                    # Container image (root, HF Spaces compliant)
├── .dockerignore                 # Docker build exclusions
├── .hfignore                     # HF Space upload exclusions
├── .gitignore                    # Git exclusions
├── __init__.py                   # Package exports (AmlEnv, AmlAction, AmlObservation)
├── client.py                     # AmlEnv WebSocket client
├── models.py                     # Pydantic action/observation schemas
├── inference.py                  # Baseline RL agent (OpenAI client, [START]/[STEP]/[END] logs)
├── openenv.yaml                  # OpenEnv manifest (tasks, graders, port)
├── pyproject.toml                # Project metadata and uv dependencies
├── uv.lock                       # Locked dependency graph
├── README.md                     # This file (also HF Space card)
├── data/
│   ├── entities.json             # 312 KYC entity records
│   ├── accounts.json             # 410 bank accounts
│   └── transactions.json         # 5,079 transactions (haystack + fraud scenarios)
├── graders/
│   ├── __init__.py
│   ├── aml_easy.py               # "The False Positive" grader
│   ├── aml_medium.py             # "The Smurf Network" grader
│   └── aml_hard.py               # "The Corporate Mirage" grader
├── server/
│   ├── __init__.py
│   ├── AML_env_environment.py    # Core OpenEnv environment (reset/step/state)
│   ├── app.py                    # FastAPI server (CORS, create_app wrapper)
│   └── requirements.txt          # Pip fallback requirements
└── tools/
    ├── haystack.py               # Financial graph generator
    └── tasks.json                # Manual fraud scenario definitions
```
