# Design choices: what I chose, what I compared, and how sure I am

Every choice is labelled by the kind of evidence behind it:

- **Measured**: I ran the alternatives on labelled data and the numbers are below.
- **Reasoned**: I picked it from the constraints of the task and did not benchmark the alternatives.
- **Not compared**: a given or default choice; no alternative was tried.

Small evaluation sets (24-36 questions, 50 reports) mean one query moves a metric by 3-4 points. Differences of that size are noise and I say so where it applies. Reproduce the measurements with `python -m scripts.eval_retrieval` and `python -m scripts.compare_choices` (the second needs `pip install -r requirements-dev.txt`).

## 1. All choices at a glance

| # | Area | Chosen | Alternatives | Evidence |
|---|---|---|---|---|
| 1 | Model and SDK | Gemini (`gemini-3.5-flash-lite`) through `google-genai`; Gemini embeddings | Claude, OpenAI models | Not compared (free-tier key I had) |
| 2 | Agent structure | Hand-written tool-calling loop (about 100 lines) | LangChain/LangGraph, Google ADK, SDK automatic function calling | Reasoned |
| 3 | Routing | LLM function calling over 3 described tools | Keyword/rule router, separate classifier call | Reasoned |
| 4 | Product retrieval | BM25 ranks pages; embedding similarity is a loose on-topic gate | Embeddings only, hybrid | Measured (2.1) |
| 5 | Product name added to each chunk | Yes | Plain page text | Measured (2.1), small gain |
| 6 | BM25 implementation | My own (about 30 lines) | `rank_bm25` package | Measured (2.2): equal |
| 7 | BM25 tokenizer | Stop-words + plural stripping | No processing, stop-words only, stemming only | Measured (2.2): chosen one is not the best, within noise |
| 8 | Chunk size | One page per chunk | Half-page chunks | Measured (2.3): equal |
| 9 | Several queries per search | Model's rewrite + user's own words, rank-fused, top 6 | Rewrite only, top 4 | Measured (2.4) |
| 10 | Wrong-answer control | LLM must quote; code verifies quote and numbers; second support check; reply guard | Plain RAG with citations; LLM judge only | Outcome measured (2.5); **baseline not yet run** |
| 11 | Service reports to table | LLM extraction + number validation, built once | Regex/rule parser | Measured (2.6) |
| 12 | Cost estimate | Plain-code bottom-up from similar visits | LLM-written number; mean of totals; ML regression | Reasoned from the data (2.7) |
| 13 | Currency | Never converted | Fixed rates, exchange-rate API | Reasoned |
| 14 | Flotation analysis | pandas percentiles; linear vs boosting; chronological split | Random split; statsmodels | Measured (2.8) |
| 15 | Libraries | numpy, pandas, scikit-learn, pydantic, python-dotenv, pytest | polars, statsmodels, attrs, unittest | Reasoned (section 3) |
| 16 | Retries, logs, evals, prompts | Own retry wrapper, redacted JSONL logs, own eval scripts, prompt files in git | tenacity, OpenTelemetry/LangSmith, ragas/DeepEval/promptfoo, prompt-management service | Reasoned (section 4) |

## 2. Measured comparisons

### 2.1 Retrieval method (36 labelled queries: 24 answerable, 12 not; 63 page chunks)

"Hit@k" = a correct page is in the top k. "AUROC" = how well the top score separates answerable from unanswerable queries (0.5 = useless, 1.0 = perfect).

| method | Hit@1 | Hit@3 | Hit@5 | MRR | Hit@3 on 7 hard queries | AUROC |
|---|---|---|---|---|---|---|
| BM25 | 0.708 | 1.000 | 1.000 | 0.854 | 1.000 | 0.611 |
| embeddings | 0.667 | 0.917 | 0.958 | 0.795 | 0.714 | 0.934 |
| hybrid (50/50) | 0.708 | 1.000 | 1.000 | 0.854 | 1.000 | 0.736 |
| **BM25 + product name** | 0.750 | **1.000** | 1.000 | **0.875** | **1.000** | 0.632 |
| embeddings + product name | 0.750 | 0.958 | 0.958 | 0.842 | 0.857 | **0.944** |
| hybrid + product name | 0.708 | 1.000 | 1.000 | 0.854 | 1.000 | 0.753 |

Reading it: BM25 finds the right page most reliably (embeddings missed the Larox demo prompt entirely, rank 23). Embedding similarity is the only score that separates answerable from unanswerable queries. Hybrid did not beat BM25 at ranking or embeddings at gating, so I use each for its strength. Adding the product name helps a little for BM25 and embeddings.
Caveats: labels written by me; the 6 extra off-topic queries and the boilerplate-footer removal were added after seeing first results.

### 2.2 BM25 implementation and tokenizer

| variant | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| my BM25 | 0.750 | 1.000 | 1.000 | 0.875 |
| `rank_bm25.BM25Okapi` | 0.750 | 1.000 | 1.000 | 0.868 |

| tokenizer | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| no stop-words, no stemming | 0.792 | 0.958 | 1.000 | 0.883 |
| stop-words only | 0.792 | 1.000 | 1.000 | 0.896 |
| stemming only | 0.667 | 0.917 | 1.000 | 0.803 |
| stop-words + plural stemming (used) | 0.750 | 1.000 | 1.000 | 0.875 |

Honest reading: the package is just as good, so writing my own was not about accuracy; it keeps the dependency list short and lets me compute a score ceiling. The tokenizer I use is not the best of these: stop-words only scores slightly higher (one query's difference) and stemming alone hurts. I kept the original because the gap is within noise, but "stop-words only" is what I would switch to.

### 2.3 Chunk size (BM25 only)

| chunking | chunks | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| whole page (used) | 63 | 0.750 | 1.000 | 1.000 | 0.875 |
| half page | 125 | 0.750 | 1.000 | 1.000 | 0.875 |

Identical, so I keep pages: simpler, and a page number is a natural citation. Not tested with embeddings.

### 2.4 Several queries per search

The model rewrites the user's message before calling the product tool, and the rewrite can drop key words. For the Larox prompt the correct page (`filter-optimizer` p.2) ranked 2nd for the user's own words but 5th for the model's rewrite ("Larox **PF** filter … feed **conditions**"), outside the top 4 pages passed to the answerer.

| setup | Larox prompt found the Filter optimizer |
|---|---|
| rewrite only, top 4 | 2 of 6 runs |
| rewrite + user's words fused (reciprocal rank fusion), top 6, extractor told to cover all relevant sources | 9 of 9 completed runs |

(A 10th run failed on the API quota, not on quality.)

### 2.5 Controlling wrong answers

Layers: loose on-topic gate, verbatim quote per claim, code check that the quote is on the cited page and every number is in the quote, second check that the quote supports the claim, answer text built only from verified claims, and a reply guard that checks every number and product name in the final reply against tool results.
Outcome on 21 development questions (missing facts, false premises, injection, conflicting brochures, flattened tables): **0 wrong answers**, 8 correct, 11 correct refusals, 2 answerable questions refused.
**Not done yet:** a baseline of plain RAG (same retrieval, answer with citations, no verification) on the same questions. Without it I can show the safety layers work, but not how many wrong answers they prevent. Planned once the API quota is available.

### 2.6 Turning the 50 service reports into a table

Layouts: 14 reports have markdown tables, 35 are short free-text notes, 1 is a long note; costs appear as "Total Due", "Grand Total", "Total cost:", "as per agreement", or not at all.
A regex parser I wrote for this comparison versus the LLM extraction, scored on the 28 values (total, on-site hours, travel hours) I read by hand from 18 reports:

| method | correct | wrong value | missed |
|---|---|---|---|
| regex parser | 21 | 5 | 2 |
| LLM extraction + number validation (used) | 28 | 0 | 0 |

The regex failures were mostly the same mistake: reading total on-site *hours* as the total cost when the cost line was worded differently, or missing a travel figure written in prose.
Caveat in the regex's favour: I wrote it once, while I tuned the LLM prompt after seeing failures on these same reports, so the gap is overstated. A stronger regex would close part of it, but the 35 free-text notes would keep it brittle.

### 2.7 Why the cost estimate is plain code over similar visits (reasoned from data)

| alternative | why not |
|---|---|
| Ask the LLM for a number | Would be invented, not "from the reports". |
| Average of all stated totals | Only 29 of 50 reports state a total; they are in 6 currencies; travel and rates differ by country. |
| Regression / ML | Only 6 similar visits (PF-DS cloth jobs), 0 of them in Chile. Too few rows to fit anything honestly. |
| Convert everything to one currency | The reports contain no exchange rates; I would be inventing them. |
| **Used:** hours from similar jobs x the target country's rates + its travel hours + part prices from similar jobs, plus the raw totals of the similar visits | Every figure traces to listed report IDs. |

Estimate for the demo prompt: labour + travel USD 734-2,546 (typical about 1,591), with parts listed separately in EUR and CAD.

### 2.8 Flotation: what the data can support

| split | linear model R2 | gradient boosting R2 |
|---|---|---|
| chronological: train on first 70% of hours, test on last 30% (used) | -0.22 | 0.07 |
| random 70/30, mean of 5 shuffles | 0.16 | 0.31 |

A random split mixes neighbouring hours (which look alike) into train and test and makes the model look useful. Testing on later data is the honest question for "would this advice work next week", and it says process settings do not predict silica (R2 below 0.3 is treated as "no firm advice"). Not compared: other models, or statsmodels for p-values (I did not need p-values; the claim is about prediction).

## 3. Libraries

| library | used for | alternative | why this one |
|---|---|---|---|
| `google-genai` | model calls, function calling, structured output, embeddings | `openai`, `anthropic` | Not compared; matched the key I had. |
| `pydantic` | validated tool arguments, extraction schemas, structured model output | dataclasses, attrs | Gives JSON schemas for the model and clear validation errors in one tool. |
| `numpy` | vectors, cosine similarity, linear algebra | pure Python | Embedding matrix products and least squares. |
| `pandas` | flotation CSV analysis | polars, plain `csv` | Percentiles, filtering and grouping are one-liners; 4,097 rows, speed is irrelevant. |
| `scikit-learn` | gradient boosting for the out-of-time check | statsmodels, xgboost | Smallest dependency that provides a non-linear baseline. |
| `python-dotenv` | read the API key from `.env` | environment variables only | Keeps the key out of the repo. |
| `pytest` | tests | unittest | Less boilerplate, fixtures for the log redirect. |
| `rank_bm25` | dev only: compare against my BM25 | n/a | Listed in `requirements-dev.txt`. |

## 4. Reasoned-only choices

| choice | alternative | reasoning |
|---|---|---|
| Own agent loop | LangChain, LangGraph, ADK, SDK automatic function calling | I needed a step limit, argument validation, slot checks, reply guard and a trace at exact points. The loop is short enough to explain line by line. A framework would be reasonable at larger scale. |
| LLM routes via tool descriptions | keywords or a classifier | Messages mix topics ("readings ... what could we suggest?") and need more than one tool; tool descriptions also tell the model when not to call. |
| Own retry wrapper in `src/llm.py` | `tenacity` | Needed one rule the library does not give: fail immediately on a daily quota instead of waiting hours. (The SDK already retries internally with tenacity for some errors.) |
| JSONL logs with redaction | OpenTelemetry, LangSmith | Enough for one process; at larger scale tracing would be added. |
| Own eval scripts | ragas, DeepEval, promptfoo | The checks I needed (which tool ran, which arguments, forbidden facts) are specific to this task and short to write. |
| Input handling (see README 2.1 for the list) | separate intent classifier, rules-only extraction, forms, confirm-everything, summarised memory | The tool-calling model routes and extracts in one step; code only checks what it extracted. |
| Prompts as versioned `.md` files with a hash in every log record | prompts in code, prompt-management service | Reviewable in git diffs, and a run can be tied to the exact prompt text. |

## 5. What I did not compare

- Other model providers or model sizes (including whether a stronger model would need fewer safety layers).
- Embedding models other than `gemini-embedding-001`, and chunk size for embeddings.
- Plain RAG without verification (planned, see 2.5).
- Other ways to read the PDFs (I used the provided text extraction, whose flattened tables are the main reason spec numbers from tables are refused or flagged).

## 6. Configuration and limits

Basis legend: **Measured** = backed by a comparison I ran; **Tuned** = chosen from the same data I report on, so it is optimistic; **Judgement** and **defaults** = not tuned. Changing any of them is a one-line edit in the file shown.

| Constant | Value | Where | Basis |
|---|---|---|---|
| **Agent and reliability** | | | |
| Model calls per user message | 6 | `MAX_STEPS`, src/agent.py | Judgement. A typical message needs 1-3; 6 leaves room for one correction (e.g. a rejected argument) without letting a looping model run on. |
| Deadline for one user message | 180 s | `TURN_DEADLINE_S`, src/config.py | Judgement. Stops a turn stuck in slow retries. |
| Timeout per model request | 60 s | `REQUEST_TIMEOUT_S` | Judgement. |
| Attempts per request, waits | 4 attempts; waits 2 / 5 / 12 s plus jitter | `MAX_ATTEMPTS`, `BACKOFF_S` | Judgement. Only temporary errors are retried; a daily-quota error fails at once. |
| **Product search** | | | |
| Pages sent to the answerer | 6 | `TOP_K`, src/config.py | Measured: a correct page was in the top 5 for all 24 answerable queries. Raised from 4 after the right page ranked 5th in 4 of 6 runs of the Larox prompt. |
| On-topic gate (embedding cosine) | 0.70 | `GATE` | Tuned on the same 36 queries from the score gap (lowest answerable 0.68-0.70, most off-topic below it). Loose on purpose: it is not the safety mechanism. |
| BM25 parameters k1 / b | 1.5 / 0.75 | src/tools/product_search.py | Common defaults, not tuned. |
| Rank-fusion constant | 60 | same file | Standard value for reciprocal rank fusion, not tuned. |
| Embedding batch size | 40 | same file | Practical: keeps each request small enough for the free-tier per-minute limit. |
| Smallest page kept | 5 words of 3+ letters | src/ingest/brochures.py | Judgement. Drops image-only pages. |
| **Answer checks** | | | |
| Shortest quote accepted as evidence | 3 words | `MIN_QUOTE_TOKENS`, src/verifier.py | Judgement. Stops one-word 'quotes'. |
| Shortest customer phrase in a recommendation | 2 words | src/grounded.py | Judgement. |
| Prompt-leak check window | 10 consecutive words | `SHINGLE`, src/guard.py | Raised from 8 after 8 flagged ordinary phrases. |
| **Input handling** | | | |
| Longest message accepted | 2,000 characters | `MAX_INPUT_CHARS`, src/config.py | Judgement. A salesperson's question is a sentence or two; this blocks accidental pastes before any model call costs tokens. |
| Size of the conversation resent | about 20,000 tokens, older whole messages dropped beyond it | `HISTORY_TOKEN_BUDGET` | Judgement. A normal call never reaches it, so it only guards very long chats. Checked details are pinned and never dropped. |
| Country typo tolerance | similarity of 0.85 or more, names of 4+ letters | src/slots.py | Judgement. Accepts a one-letter typo such as "Chille". |
| **Service cost** | | | |
| Minimum similar past visits for an estimate | 3 | `MIN_SIMILAR`, src/tools/service_cost.py | Judgement. Below 3 a min-max range says almost nothing. |
| **Flotation analysis** | | | |
| Train / test split | first 70% of hours / last 30% | src/tools/flotation.py | Measured: a random split overstates R2 (0.16-0.31 vs -0.22-0.07). |
| R2 needed before settings count as predictive | 0.30 | `TRUSTED_R2` | Judgement. |
| Time blocks for the stability check | 5 | same file | Judgement. |
| Gradient boosting | 200 trees, depth 3 | same file | Typical starting values, not tuned. |
| Neighbourhood of the customer's reading | iron +/-0.25, silica +/-0.15 | same file | Judgement. Narrow enough to be specific; gives about 490 and 220 hours. |
| 'Recent' window | 30 days | same file | Judgement. |
| High vs low silica comparison | silica >= 3.1 vs < 2.0 (fixed) | same file | Known weakness: fixed to the example prompt, not to the customer's reading. It should follow the input. |
| **Cost** | | | |
| Price per million tokens | not set | `PRICE_PER_MILLION_TOKENS` | Left empty on purpose: no verified price, so no dollar figures. |

Because most of these are judgement calls on small test sets, treat them as sensible starting points and not as optimal values.
