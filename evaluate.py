import time
from rouge_score import rouge_scorer
from rag_chain import get_rag_chain, get_retriever

# -------------------------------------------------------------------
# Test set: (question, expected_answer, keyword_in_relevant_chunk)
# Replace these with questions relevant to your actual PDF
# -------------------------------------------------------------------
TEST_CASES = [
    {
        "question": "How many days of leave are employees entitled to per year?",
        "expected": "12 days of leave per annum",
        "relevant_keyword": "leave",
    },
    {
        "question": "What is the notice period for resignation?",
        "expected": "30 days notice period",
        "relevant_keyword": "notice",
    },
    {
        "question": "What is the probation period for new employees?",
        "expected": "6 months probation period",
        "relevant_keyword": "probation",
    },
]

def measure_response_quality(chain, test_cases):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    results = []
    for tc in test_cases:
        response = chain.invoke(tc["question"])
        scores = scorer.score(tc["expected"], response)
        results.append({
            "question": tc["question"],
            "rouge1": round(scores["rouge1"].fmeasure, 3),
            "rougeL": round(scores["rougeL"].fmeasure, 3),
            "response": response,
        })
    return results


def measure_retrieval_quality(retriever, test_cases):
    results = []
    for tc in test_cases:
        docs = retriever.invoke(tc["question"])
        hits = sum(1 for d in docs if tc["relevant_keyword"].lower() in d.page_content.lower())
        precision = round(hits / len(docs), 3) if docs else 0
        results.append({
            "question": tc["question"],
            "chunks_retrieved": len(docs),
            "relevant_hits": hits,
            "precision@k": precision,
        })
    return results


def measure_performance(chain, test_cases):
    results = []
    for tc in test_cases:
        start = time.time()
        chain.invoke(tc["question"])
        elapsed = round(time.time() - start, 2)
        results.append({"question": tc["question"], "response_time_sec": elapsed})
    return results


def print_section(title, rows, keys):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    for row in rows:
        print(f"\n  Q: {row['question']}")
        for k in keys:
            print(f"     {k}: {row[k]}")


if __name__ == "__main__":
    print("Loading chain and retriever...")
    chain = get_rag_chain()
    retriever = get_retriever()

    rq = measure_response_quality(chain, TEST_CASES)
    print_section("RESPONSE QUALITY (ROUGE)", rq, ["rouge1", "rougeL", "response"])

    ret = measure_retrieval_quality(retriever, TEST_CASES)
    print_section("RETRIEVAL QUALITY (Precision@K)", ret, ["chunks_retrieved", "relevant_hits", "precision@k"])

    perf = measure_performance(chain, TEST_CASES)
    print_section("PERFORMANCE (Response Time)", perf, ["response_time_sec"])

    avg_time = round(sum(p["response_time_sec"] for p in perf) / len(perf), 2)
    avg_rouge1 = round(sum(r["rouge1"] for r in rq) / len(rq), 3)
    avg_precision = round(sum(r["precision@k"] for r in ret) / len(ret), 3)
    print(f"\n{'='*60}")
    print(f" SUMMARY")
    print(f"{'='*60}")
    print(f"  Avg ROUGE-1:      {avg_rouge1}")
    print(f"  Avg Precision@K:  {avg_precision}")
    print(f"  Avg Response Time: {avg_time}s")
