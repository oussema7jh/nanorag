# nanorag

[![CI](https://github.com/oussema7jh/nanorag/actions/workflows/ci.yml/badge.svg)](https://github.com/oussema7jh/nanorag/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-00f0ff?logo=python&logoColor=white)](https://www.python.org)
[![Dependencies](https://img.shields.io/badge/dependencies-0-ff2bd6)](#why-from-scratch)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

**Retrieval-Augmented Generation, built from scratch.** No LangChain, no vector-DB-as-a-service,
no API keys required. Every layer of a production RAG stack — chunking, embeddings, vector
storage, hybrid retrieval, answer generation, evaluation — implemented in pure Python and
wired together behind a small CLI.

```
┌─────────────┐   chunk    ┌─────────────┐   embed    ┌──────────────┐
│  documents  │──────────►│   chunks    │──────────►│ vector store │
│  (*.md/txt) │            │ (+ overlap) │            │  (+ BM25)    │
└─────────────┘            └─────────────┘            └──────┬───────┘
                                                             │ retrieve
                             ┌──────────────┐   RRF fusion    │
                             │   passages   │◄────────────────┘
                             │  ranked list │
                             └──────┬───────┘
                                    │ generate (grounded + cited)
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
              extractive         OpenAI /        evaluation
              (offline)       Anthropic/Ollama   recall@k · MRR
```

## Why from scratch?

Frameworks are great — once you know what they hide. This project is the
version of RAG where nothing is hidden:

| Layer | What it does | The from-scratch bit |
|---|---|---|
| **Chunking** | recursive splitting, overlap, heading context | [nanorag/chunking.py](nanorag/chunking.py) |
| **Embeddings** | feature-hashing over word + char n-grams | [nanorag/embeddings.py](nanorag/embeddings.py) |
| **Vector store** | cosine search, JSON persistence, idempotent ingest | [nanorag/store.py](nanorag/store.py) |
| **Retrieval** | Okapi BM25 + dense search fused with **RRF** | [nanorag/retrieval.py](nanorag/retrieval.py) |
| **Generation** | grounded answering with `[n]` citations — offline or LLM-backed | [nanorag/providers.py](nanorag/providers.py) |
| **Evaluation** | golden-set recall@k, MRR, citation rate | [nanorag/evals.py](nanorag/evals.py) |

**Zero runtime dependencies.** `pip install` nothing — clone and run.

## Quickstart

```bash
git clone https://github.com/oussema7jh/nanorag.git
cd nanorag

# no install step needed — but if you like:
pip install -e .          # gives you the `nanorag` console script

# 1. ingest the bundled "Night City Archives" corpus
python -m nanorag ingest docs/corpus

# 2. ask questions (works fully offline)
python -m nanorag ask "What is ICE and why is it dangerous?" -s

# 3. run the evaluation suite
python -m nanorag eval

# 4. interactive chat
python -m nanorag chat
```

`ask` output (offline extractive provider, real output from this repo):

```
$ python -m nanorag ask "What are the four phases of the Sandbreak protocol?"

answer [extractive] (3 ms)
The standard quiet-breach sequence taught in underground academies is called
the Sandbreak protocol. [1] Its four phases are: scan the subnet without
touching it, spoof a legitimate handshake, deploy the breach daemon against
the weakest ICE segment, and extract before the countermeasures recalibrate. [1]
```

## Evaluation

Golden set: 20 questions over 6 documents (`evals/golden.jsonl`), run in CI on
every push ([workflow](.github/workflows/ci.yml)):

```
$ python -m nanorag eval

QUESTION                                     RANK  CITED
--------------------------------------------------------
What is ICE and why is it dangerous?            1    yes
What does black ICE do to intruders?            1    yes
What are the four phases of the Sandbreak...    1    yes
...
--------------------------------------------------------
recall@5: 100%   MRR: 0.975   citation rate: 100%   (20 cases)
```

* **recall@5** — the expected document appears in the top-5 retrieved passages
* **MRR** — mean reciprocal rank of the first correct passage
* **citation rate** — the generated answer cites a passage from the expected document

Retrieval metrics are provider-independent; citation rate measures the full
retrieve → generate path. When an eval regresses, the table shows exactly
which questions got worse.

## Design decisions

**Hybrid retrieval (BM25 + vectors, fused with RRF).** Pure lexical search misses
paraphrases; pure dense search misses exact identifiers. Reciprocal Rank Fusion
combines both ranked lists without needing comparable scores — a standard
production technique, implemented in ~20 lines in
[retrieval.py](nanorag/retrieval.py).

**Feature-hashing embeddings.** The default embedder hashes word and character
n-grams into a fixed 512-dim space with signed accumulation (Weinberger et al.,
2009), then L2-normalises. It's deterministic, needs no model download, and
gives the dense leg of hybrid retrieval a real lexical-similarity signal. The
`Embedder` protocol makes swapping in a neural embedder a one-class change —
an `OpenAIEmbedder` adapter is included.

**Chunk overlap.** Sentences cut at a chunk boundary stay retrievable because
each chunk carries a tail of the previous one ([chunking.py](nanorag/chunking.py)).

**Offline-first generation.** The default `ExtractiveProvider` composes answers
from retrieved sentences using IDF-weighted term overlap — no API key, no cost,
fully deterministic, and it always cites. Set `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY` or `OLLAMA_URL` (or pass `--provider`) to route the same
retrieved context through a real chat model with the same citation contract.

**Document replacement.** Re-ingesting a file replaces its previous passages,
including when the file becomes empty. Storage IDs include a hash of the resolved
source path, so equally named files in different directories remain distinct.
Unchanged files require no new embeddings. Both keyword and vector retrieval
refresh after replacement, even when the number of chunks stays the same.
Replacement embeddings are computed before removing a file's existing passages.

```bash
# After editing your local Markdown/text knowledge base, run ingest again:
python -m nanorag ingest docs/corpus
```

The library result includes `new_chunks` (all passages inserted for replaced
files) and `removed_chunks`. Existing indexes migrate each file on its next
ingest. Files deleted from disk are not automatically pruned; ingesting an empty
file clears its indexed passages. Source identity uses absolute resolved paths,
so moving a corpus creates new identities. Replacement is atomic per file in
memory, not a transaction across an entire directory; concurrent writers are
not supported. See the [delivery roadmap](docs/ROADMAP.md).

## Project layout

```
nanorag/
├── nanorag/
│   ├── __init__.py      # public API
│   ├── chunking.py      # recursive chunker
│   ├── embeddings.py    # hashed n-gram embedder (+ OpenAI adapter)
│   ├── store.py         # persistent vector store
│   ├── retrieval.py     # BM25 / vector / hybrid RRF
│   ├── providers.py     # extractive + OpenAI + Anthropic + Ollama
│   ├── pipeline.py      # ingest → retrieve → generate
│   ├── evals.py         # golden-set evaluation
│   └── cli.py           # argparse CLI
├── tests/               # 31 unit + integration tests
├── docs/corpus/         # demo knowledge base (6 documents)
├── evals/golden.jsonl   # labelled question set
└── .github/workflows/   # CI: tests + ingest + eval + smoke ask
```

## Using it as a library

```python
from nanorag import RAGPipeline

pipe = RAGPipeline.open(data_dir=".nanorag")
pipe.ingest_path("docs/corpus")

answer = pipe.ask("What is the Witness Chain?")
print(answer.text)                    # grounded, cited answer
print([p.source for p in answer.passages])  # where it came from
```

## Roadmap

- [x] Safe re-ingestion of edited/empty files and distinct same-name sources
- [ ] Explicit, scoped removal of deleted source files
- [ ] Practical support-document corpus and answerability evaluation

- [ ] Sentence-transformers embedder behind the `Embedder` protocol
- [ ] SQLite-backed store for larger corpora
- [ ] Query rewriting / multi-query retrieval
- [ ] Streaming answers from chat-model providers

## License

[MIT](LICENSE) © Oussema Braham
