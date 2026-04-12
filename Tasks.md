1. The Environment Setup: The Noisy Haystack

Instead of 15 nodes, your mock database will now simulate a slice of a real bank with hundreds of entities and thousands of transactions.

The Noise (False Flags & Legitimate Activity):

    Payroll & Utilities: 80% of transactions are just companies paying hundreds of employees or paying server hosting bills.

    The "False Positive" Trap: An account transfers $12.50 to a coffee shop whose owner happens to share the exact name as someone on an international watchlist. The LLM must not waste its API budget investigating a coffee purchase.

    High Fan-Out: When the agent queries an active corporate account, it won't get 3 transactions; it will get 150.

2. The Tool Upgrades (Pagination is Mandatory)

Because the dataset is now massive, if the agent queries a major corporate account, returning 150 transactions will instantly blow up the context window and cause an Out-Of-Memory (OOM) failure for the LLM. You must introduce pagination to the tools.

    query_transactions(account_id: str, limit: int = 10, offset: int = 0): The agent must actively choose to page through history or search by date ranges.

    search_transactions_by_keyword(account_id: str, keyword: str): Allows the agent to search the "notes" or "memo" field (e.g., searching for "consulting" or "invoice") to filter noise.

    get_kyc_record(entity_id: str): Returns corporate directors, registration addresses, and incorporation dates.

    submit_decision(decision: Literal["FRAUD", "CLEAR"], evidence_links: List[str]): The final action.

    
Task 1: The False Positive Triage (Easy)

    The Alert: The automated system flags ACC-101 (A local construction company) because it transferred $50,000 to ACC-909 (A newly registered entity in a high-risk jurisdiction).

    The Environment State: ACC-101 has 200 normal transactions (buying lumber, paying contractors). The $50,000 transfer has the memo "Heavy Machinery Purchase - Unit 4".

    The Trap: A naive model will see "High-Risk Jurisdiction," panic, and flag it as FRAUD.

    The Solution Path: 1.  The agent calls query_transactions('ACC-101').
    2.  Reads the memo indicating a machinery purchase.
    3.  Calls get_kyc_record('ACC-909') and sees it is registered as "Global Tractor Sales Ltd."
    4.  Calls query_transactions('ACC-909') and sees 50 other legitimate inbound payments from global construction firms.
    5.  Agent Action: Calls submit_decision("CLEAR"). The agent proves it can dismiss noise.

Task 2: The Smurf Network (Medium)

    The Alert: The system flags ACC-200 (A used car dealership) for a sudden spike in cash deposits over a 5-day period.

    The Environment State: The dealership has hundreds of regular car sale transactions. However, mixed into the ledger are 14 incoming cash deposits, all for exactly $9,900 or $9,500.

    The Trap: The LLM must notice that these deposits are suspiciously just below the standard $10,000 AML reporting threshold.

    The Solution Path:

        The agent pages through query_transactions('ACC-200') and spots the pattern of $9,900 deposits.

        The agent notes the senders: ACC-301, ACC-302, ACC-303 (The Smurfs).

        The agent queries the KYC for those three accounts.

        The KYC reveals all three accounts were opened on the exact same day, and they all list their occupation as "Student."

        Agent Action: Calls submit_decision("FRAUD", ["ACC-301", "ACC-302", "ACC-303"]). The agent proves it can identify data anomalies (structuring) without a direct loop.

Task 3: The Ultimate Corporate Mirage (Hard)

    The Alert: The system flags a $2.5 million transfer from ACC-500 (A major logistics firm) to ACC-700 (A generic consulting agency).

    The Environment State: This is the massive haystack. ACC-500 has 500+ transactions. ACC-700 also has hundreds of transactions paying out to various vendors, charities, and payroll.

    The Trap: ACC-500 also sends $100 to an entity named "Al-Qaeda Watchlist Target" (A complete false flag designed to bait the LLM into wasting its API budget).

    The Solution Path:

        Agent ignores the $100 false flag and focuses on the $2.5M transfer.

        Agent queries ACC-700 (The consultant) and uses pagination or search tools to filter out payroll.

        Agent finds that exactly 48 hours after receiving the $2.5M, ACC-700 transferred $2.4M to ACC-888 (An offshore holding company).

        Agent queries get_kyc_record('ACC-888'). The director is listed as "Robert House."

        Agent queries get_kyc_record('ACC-500') (The original sender). The director is listed as "Apex Management Corp."

        Agent queries get_kyc_record('Apex Management Corp'). The CEO of Apex is "Robert House."

        Agent Action: Calls submit_decision("FRAUD", ["ACC-500", "ACC-700", "ACC-888"]). The agent proves it can ignore severe false flags, navigate pagination, and cross-reference multi-hop KYC records to find the hidden loop.