"""Classify Four Failure Traces -- starter.

RAG fails in four layers. Each layer has a different first diagnostic,
so the label is the thing that tells you where to look first.

    retrieval   the chunk with the answer never came back
    context     it came back, but never reached the model intact
    generation  it reached the model, and the model ignored it
    operations  the answer is right -- the system around it is not
"""

LABELS = ("retrieval", "context", "generation", "operations")

LATENCY_BUDGET_S = 3.0     # anything slower than this is not shippable
COST_BUDGET_USD = 0.01     # per query

# --- The four traces --------------------------------------------------------
# gold_chunk : the chunk id that actually holds the answer (a direct keyword
#              search of the corpus found it -- this is ground truth, not a guess)
# gold_fact  : the string a correct answer must contain
# in_context : chunk ids that reached the model COMPLETE. A chunk cut in half
#              by truncation is not in this list.

TRACE_A = {
    "question": "What was the operating margin for the cloud segment?",
    "gold_chunk": "report#p15", "gold_fact": "31%",
    "retrieved": [{"id": "report#p12", "score": 0.81, "text": "cloud demand was strong across enterprise customers"},
                  {"id": "report#p14", "score": 0.79, "text": "segment performance is discussed in the MD&A section"},
                  {"id": "report#p09", "score": 0.77, "text": "the table below summarizes income by segment"}],
    "in_context": ["report#p12", "report#p14", "report#p09"], "truncated": False,
    "answer": "Cloud saw strong enterprise demand; the margin is discussed in the MD&A section.",
    "latency_s": 1.8, "cost_usd": 0.004, "index_stale": False,
}

TRACE_B = {
    "question": "What is the retention period for customer support recordings?",
    "gold_chunk": "privacy#s2", "gold_fact": "90 days",
    "retrieved": [{"id": "hr#s3", "score": 0.74, "text": "employee records are retained for seven years"},
                  {"id": "hr#s4", "score": 0.73, "text": "retention of performance documentation follows"},
                  {"id": "privacy#s2", "score": 0.72, "text": "support call recordings are retained for 90 days"}],
    "in_context": ["hr#s3", "hr#s4"], "truncated": True,   # privacy#s2 was cut mid-chunk at 800 tokens
    "answer": "Records are retained for seven years.",
    "latency_s": 2.1, "cost_usd": 0.005, "index_stale": False,
}

TRACE_C = {
    "question": "How many regional offices does the company operate?",
    "gold_chunk": "handbook#p04", "gold_fact": "14",
    "retrieved": [{"id": "handbook#p04", "score": 0.88, "text": "the company operates 14 regional offices"},
                  {"id": "handbook#p05", "score": 0.82, "text": "regional office leadership reports into the COO"},
                  {"id": "handbook#p31", "score": 0.75, "text": "office locations are listed in Appendix C"}],
    "in_context": ["handbook#p04", "handbook#p05", "handbook#p31"], "truncated": False,
    "answer": "The company operates approximately 20 regional offices worldwide.",
    "latency_s": 1.6, "cost_usd": 0.004, "index_stale": False,
}

TRACE_D = {
    "question": "What is the current parental leave policy?",
    "gold_chunk": "hr#s7", "gold_fact": "12 weeks",
    "retrieved": [{"id": "hr#s7", "score": 0.91, "text": "eligible employees may take up to 12 weeks of paid parental leave"},
                  {"id": "hr#s8", "score": 0.84, "text": "leave may be taken continuously or intermittently"},
                  {"id": "benefits#p06", "score": 0.79, "text": "parental leave benefits are administered through HR"}],
    "in_context": ["hr#s7", "hr#s8", "benefits#p06"], "truncated": False,
    "answer": "Eligible employees may take up to 12 weeks of paid parental leave. [hr#s7]",
    "latency_s": 11.4, "cost_usd": 0.021, "index_stale": True,   # source now says 16 weeks; index built in March
}

TRACES = {"A": TRACE_A, "B": TRACE_B, "C": TRACE_C, "D": TRACE_D}


def classify(trace: dict) -> str:
    """Return the layer that failed first: "retrieval", "context", "generation" or "operations".

    TODO: derive one rule per layer from the evidence in the trace and check them
    in pipeline order (retrieval -> context -> generation -> operations).

    TODO: Trace B is ambiguous. State here which label you chose for it and why.
    """
    retrieved_ids = [chunk["id"] for chunk in trace["retrieved"]]

    # TODO Rule 1 -- retrieval: is the gold chunk missing from retrieved_ids?

    # TODO Rule 2 -- context: was it retrieved but missing from in_context?

    # TODO Rule 3 -- generation: did it reach the model but the answer lacks gold_fact?

    # TODO Rule 4 -- operations: the answer is right, so look at the system around it.

    return ""


if __name__ == "__main__":
    for name, trace in TRACES.items():
        print(f"Trace {name}: {classify(trace) or '(not classified yet)'}")
