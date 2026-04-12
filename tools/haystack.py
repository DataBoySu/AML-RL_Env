"""
Financial Knowledge Graph - Haystack Generator
================================================
Generates entities.json, accounts.json, and transactions.json
for an AI Reinforcement Learning environment.

Usage:
    python generate_haystack.py                          # Generate haystack only
    python generate_haystack.py --inject manual_tasks.json  # Inject manual fraud tasks
"""

import json
import random
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from faker import Faker

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# ── Volume constants ──────────────────────────────────────────────────────────
NUM_ENTITIES     = 300
NUM_ACCOUNTS     = 400
NUM_TRANSACTIONS = 5_000
PCT_INDIVIDUAL   = 0.80   # 80 % Individual, 20 % Corporate
PCT_ACTIVE       = 0.95   # 95 % Active accounts

# ── Time window ───────────────────────────────────────────────────────────────
NOW       = datetime.now(timezone.utc)
SIX_MONTHS_AGO = NOW - timedelta(days=182)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – PROCEDURAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def make_entity_id(n: int) -> str:
    return f"ENT-{n:04d}"

def make_account_id(n: int) -> str:
    return f"ACC-{n:04d}"

def make_txn_id(n: int) -> str:
    return f"TXN-{n:06d}"


def generate_entities(count: int = NUM_ENTITIES) -> list[dict]:
    """Generate Individual and Corporate entities."""
    entities: list[dict] = []
    n_individual = round(count * PCT_INDIVIDUAL)

    # ── Individuals first (needed as directors for Corporates) ───────────────
    for i in range(1, n_individual + 1):
        entities.append({
            "entity_id":            make_entity_id(i),
            "name":                 fake.name(),
            "type":                 "Individual",
            "registration_address": fake.address().replace("\n", ", "),
            "directors":            [],
        })

    individual_ids = [e["entity_id"] for e in entities]

    # ── Corporates ────────────────────────────────────────────────────────────
    for i in range(n_individual + 1, count + 1):
        num_directors = random.randint(1, 3)
        directors = random.sample(individual_ids, k=num_directors)
        entities.append({
            "entity_id":            make_entity_id(i),
            "name":                 fake.company(),
            "type":                 "Corporate",
            "registration_address": fake.address().replace("\n", ", "),
            "directors":            directors,
        })

    return entities


def generate_accounts(entities: list[dict], count: int = NUM_ACCOUNTS) -> list[dict]:
    """Assign accounts randomly to entities."""
    entity_ids = [e["entity_id"] for e in entities]
    accounts: list[dict] = []

    for i in range(1, count + 1):
        accounts.append({
            "account_id":      make_account_id(i),
            "owner_entity_id": random.choice(entity_ids),
            "status":          "Active" if random.random() < PCT_ACTIVE else "Closed",
        })

    return accounts


# ── Memo / amount helpers ─────────────────────────────────────────────────────

def _corp_to_individual_tx() -> tuple[str, float]:
    memo = random.choice([
        "Payroll", "Salary Q3", "Salary Q4", "Expense Reimbursement",
        "Bonus Payment", "Contractor Fee", "Freelance Invoice",
    ])
    amount = round(random.uniform(2_000, 10_000), 2)
    return memo, amount


def _corp_to_corp_tx() -> tuple[str, float]:
    memo = random.choice([
        "Server Hosting", "Consulting Retainer", "Office Supplies",
        f"Invoice #{random.randint(1000, 9999)}", "Software License",
        "Marketing Services", "Logistics Fee", "Partnership Distribution",
    ])
    amount = round(random.uniform(500, 50_000), 2)
    return memo, amount


def _individual_to_corp_tx() -> tuple[str, float]:
    memo = random.choice([
        "Utility Bill", "Coffee", "Gym Membership", "Online Shopping",
        "Streaming Subscription", "Insurance Premium", "Rent Payment",
    ])
    amount = round(random.uniform(5, 200), 2)
    return memo, amount


def _individual_to_individual_tx() -> tuple[str, float]:
    memo = random.choice([
        "Dinner split", "Birthday gift", "Loan repayment", "Shared expenses",
        "Concert tickets", "Rent share", "",
    ])
    amount = round(random.uniform(10, 500), 2)
    return memo, amount


def _random_timestamp() -> str:
    delta_seconds = (NOW - SIX_MONTHS_AGO).total_seconds()
    rand_seconds  = random.uniform(0, delta_seconds)
    ts = SIX_MONTHS_AGO + timedelta(seconds=rand_seconds)
    return ts.isoformat()


def generate_transactions(
    accounts:  list[dict],
    entities:  list[dict],
    count:     int = NUM_TRANSACTIONS,
    id_offset: int = 0,
) -> list[dict]:
    """Generate semantically-typed transactions between accounts."""

    # Build a lookup: account_id → entity type
    entity_type: dict[str, str] = {e["entity_id"]: e["type"] for e in entities}
    acct_to_entity: dict[str, str] = {
        a["account_id"]: a["owner_entity_id"] for a in accounts
    }

    active_accounts = [a["account_id"] for a in accounts if a["status"] == "Active"]
    if len(active_accounts) < 2:
        raise ValueError("Not enough active accounts to generate transactions.")

    transactions: list[dict] = []

    for i in range(1, count + 1):
        sender_acct   = random.choice(active_accounts)
        receiver_acct = random.choice(active_accounts)
        while receiver_acct == sender_acct:
            receiver_acct = random.choice(active_accounts)

        sender_type   = entity_type.get(acct_to_entity.get(sender_acct,   ""), "Individual")
        receiver_type = entity_type.get(acct_to_entity.get(receiver_acct, ""), "Individual")

        if sender_type == "Corporate" and receiver_type == "Individual":
            memo, amount = _corp_to_individual_tx()
        elif sender_type == "Corporate" and receiver_type == "Corporate":
            memo, amount = _corp_to_corp_tx()
        elif sender_type == "Individual" and receiver_type == "Corporate":
            memo, amount = _individual_to_corp_tx()
        else:
            memo, amount = _individual_to_individual_tx()

        transactions.append({
            "txn_id":          make_txn_id(i + id_offset),
            "sender_account":  sender_acct,
            "receiver_account": receiver_acct,
            "amount":          amount,
            "timestamp":       _random_timestamp(),
            "memo_text":       memo,
        })

    return transactions


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – INTEGRATION HOOKS
# ═══════════════════════════════════════════════════════════════════════════════

def _camouflage_transactions(
    manual_account_ids:    list[str],
    haystack_account_ids:  list[str],
    entities:              list[dict],
    accounts:              list[dict],
    txn_id_start:          int,
) -> list[dict]:
    """
    Generate 5-10 'normal' transactions that bridge each manual account to
    random haystack accounts, so manual accounts don't appear as isolated islands.
    """
    entity_type: dict[str, str] = {e["entity_id"]: e["type"] for e in entities}
    acct_to_entity: dict[str, str] = {
        a["account_id"]: a["owner_entity_id"] for a in accounts
    }

    camouflage: list[dict] = []
    counter = txn_id_start

    for manual_acct in manual_account_ids:
        n_bridge_txns = random.randint(5, 10)
        for _ in range(n_bridge_txns):
            haystack_acct = random.choice(haystack_account_ids)

            # Randomly decide direction (manual sends or receives)
            if random.random() < 0.5:
                sender_acct, receiver_acct = manual_acct, haystack_acct
            else:
                sender_acct, receiver_acct = haystack_acct, manual_acct

            sender_type   = entity_type.get(acct_to_entity.get(sender_acct,   ""), "Individual")
            receiver_type = entity_type.get(acct_to_entity.get(receiver_acct, ""), "Individual")

            if sender_type == "Corporate" and receiver_type == "Individual":
                memo, amount = _corp_to_individual_tx()
            elif sender_type == "Corporate" and receiver_type == "Corporate":
                memo, amount = _corp_to_corp_tx()
            elif sender_type == "Individual" and receiver_type == "Corporate":
                memo, amount = _individual_to_corp_tx()
            else:
                memo, amount = _individual_to_individual_tx()

            camouflage.append({
                "txn_id":           make_txn_id(counter),
                "sender_account":   sender_acct,
                "receiver_account": receiver_acct,
                "amount":           amount,
                "timestamp":        _random_timestamp(),
                "memo_text":        memo,
                "_camouflage":      True,   # debug tag – remove if unwanted
            })
            counter += 1

    return camouflage


def inject_manual_tasks(
    haystack_data:    dict,
    manual_json_path: str | Path,
) -> dict:
    """
    Load hand-written manual_tasks.json and merge it into the haystack.

    Expected manual_tasks.json schema:
    {
      "entities":     [ { ...entity fields... }, ... ],
      "accounts":     [ { ...account fields... }, ... ],
      "transactions": [ { ...transaction fields... }, ... ]
    }

    Returns the merged dataset dict.
    """
    manual_path = Path(manual_json_path)
    if not manual_path.exists():
        raise FileNotFoundError(f"Manual tasks file not found: {manual_path}")

    with manual_path.open() as fh:
        manual: dict = json.load(fh)

    # ── Validate top-level keys ───────────────────────────────────────────────
    for key in ("entities", "accounts", "transactions"):
        if key not in manual:
            raise ValueError(
                f"manual_tasks.json is missing the '{key}' key. "
                "Please check the expected schema in the docstring."
            )

    print(f"  Injecting {len(manual['entities'])} manual entities …")
    print(f"  Injecting {len(manual['accounts'])} manual accounts …")
    print(f"  Injecting {len(manual['transactions'])} manual transactions …")

    # ── Collect IDs already in the haystack to detect collisions ─────────────
    existing_entity_ids = {e["entity_id"] for e in haystack_data["entities"]}
    existing_acct_ids   = {a["account_id"] for a in haystack_data["accounts"]}
    existing_txn_ids    = {t["txn_id"]     for t in haystack_data["transactions"]}

    for e in manual["entities"]:
        if e["entity_id"] in existing_entity_ids:
            raise ValueError(
                f"Collision: entity_id '{e['entity_id']}' already exists in the haystack. "
                "Use IDs outside the ENT-0001 … ENT-0300 range (e.g. ENT-9001)."
            )

    for a in manual["accounts"]:
        if a["account_id"] in existing_acct_ids:
            raise ValueError(
                f"Collision: account_id '{a['account_id']}' already exists in the haystack. "
                "Use IDs outside the ACC-0001 … ACC-0400 range (e.g. ACC-9001)."
            )

    for t in manual["transactions"]:
        if t["txn_id"] in existing_txn_ids:
            raise ValueError(
                f"Collision: txn_id '{t['txn_id']}' already exists in the haystack. "
                "Use IDs outside the TXN-000001 … TXN-005000 range (e.g. TXN-900001)."
            )

    # ── Append manual data ────────────────────────────────────────────────────
    # Build a combined entity + account list so camouflage txns can look up types
    combined_entities = haystack_data["entities"] + manual["entities"]
    combined_accounts = haystack_data["accounts"] + manual["accounts"]

    haystack_data["entities"]     += manual["entities"]
    haystack_data["accounts"]     += manual["accounts"]
    haystack_data["transactions"] += manual["transactions"]

    # ── Camouflage: bridge manual accounts to haystack accounts ──────────────
    manual_acct_ids   = [a["account_id"] for a in manual["accounts"]]
    haystack_acct_ids = [
        a["account_id"]
        for a in haystack_data["accounts"]
        if a["account_id"] not in set(manual_acct_ids) and a["status"] == "Active"
    ]

    txn_id_start = (
        max(
            int(t["txn_id"].split("-")[1])
            for t in haystack_data["transactions"]
        ) + 1
    )

    camouflage = _camouflage_transactions(
        manual_account_ids   = manual_acct_ids,
        haystack_account_ids = haystack_acct_ids,
        entities             = combined_entities,
        accounts             = combined_accounts,
        txn_id_start         = txn_id_start,
    )

    print(f"  Generated {len(camouflage)} camouflage transactions …")
    haystack_data["transactions"] += camouflage

    return haystack_data


# ═══════════════════════════════════════════════════════════════════════════════
# I/O HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _write_json(obj: list | dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    print(f"  ✓  Wrote {len(obj):,} records → {path}")


def save_dataset(data: dict, output_dir: Path = Path(".")) -> None:
    """Write the three JSON files to output_dir."""
    _write_json(data["entities"],     output_dir / "entities.json")
    _write_json(data["accounts"],     output_dir / "accounts.json")
    _write_json(data["transactions"], output_dir / "transactions.json")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a financial Knowledge Graph haystack."
    )
    parser.add_argument(
        "--inject",
        metavar="MANUAL_JSON",
        help="Path to hand-written manual_tasks.json to inject into the haystack.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory where the three JSON files will be written (default: cwd).",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # ── 1. Build the haystack ─────────────────────────────────────────────────
    print("\n── Step 1: Generating entities …")
    entities = generate_entities()
    print(f"  ✓  {len(entities)} entities "
          f"({sum(1 for e in entities if e['type']=='Individual')} individuals, "
          f"{sum(1 for e in entities if e['type']=='Corporate')} corporates)")

    print("── Step 2: Generating accounts …")
    accounts = generate_accounts(entities)
    print(f"  ✓  {len(accounts)} accounts "
          f"({sum(1 for a in accounts if a['status']=='Active')} active)")

    print("── Step 3: Generating transactions …")
    transactions = generate_transactions(accounts, entities)
    print(f"  ✓  {len(transactions):,} transactions")

    dataset: dict = {
        "entities":     entities,
        "accounts":     accounts,
        "transactions": transactions,
    }

    # ── 2. Optionally inject manual fraud tasks ───────────────────────────────
    if args.inject:
        print(f"\n── Injecting manual tasks from: {args.inject}")
        dataset = inject_manual_tasks(dataset, args.inject)

    # ── 3. Write outputs ──────────────────────────────────────────────────────
    print(f"\n── Writing JSON files to: {output_dir.resolve()}")
    save_dataset(dataset, output_dir)

    print("\n✅  Done.\n")
    print("  Dataset summary:")
    print(f"    Entities:     {len(dataset['entities']):>6,}")
    print(f"    Accounts:     {len(dataset['accounts']):>6,}")
    print(f"    Transactions: {len(dataset['transactions']):>6,}")


if __name__ == "__main__":
    main()