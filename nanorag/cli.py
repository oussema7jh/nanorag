"""Command-line interface: ingest, ask, chat, eval, info."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .evals import evaluate, load_golden
from .pipeline import DEFAULT_DATA_DIR, RAGPipeline
from .providers import resolve_provider

CYAN, DIM, RESET = "\033[96m", "\033[2m", "\033[0m"


def _tty() -> bool:
    return sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if _tty() else text


def _print_sources(answer) -> None:
    for p in answer.passages:
        loc = p.source or p.chunk.doc_id
        if p.chunk.section:
            loc += f" — {p.chunk.section}"
        print(_c(f"  [{p.rank}] {loc} (score {p.score:.4f})", DIM))
        print(_c(f"      {p.chunk.text[:100]}...", DIM))


def _open_pipeline(args) -> RAGPipeline:
    provider = resolve_provider(getattr(args, "provider", None))
    pipeline = RAGPipeline.open(data_dir=args.data_dir)
    pipeline.provider = provider
    return pipeline


# ---------------------------------------------------------------------- #
# commands
# ---------------------------------------------------------------------- #


def cmd_ingest(args) -> int:
    pipeline = _open_pipeline(args)
    stats = pipeline.ingest_path(args.path, patterns=args.patterns)
    print(
        f"indexed {stats['documents']} documents "
        f"({stats['new_chunks']} new chunks, {stats['chunks']} total) "
        f"-> {pipeline.index_path}"
    )
    if stats["documents"] == 0:
        print(_c("no matching files found (default patterns: *.md, *.txt)", DIM))
        return 1
    return 0


def cmd_ask(args) -> int:
    pipeline = _open_pipeline(args)
    answer = pipeline.ask(args.question, k=args.k)
    print(_c(f"answer [{answer.provider}] ({answer.latency_ms:.0f} ms)", CYAN))
    print(answer.text)
    if args.show_sources:
        print(_c("sources:", CYAN))
        _print_sources(answer)
    return 0


def cmd_chat(args) -> int:
    pipeline = _open_pipeline(args)
    print(_c(f"nanorag chat — provider: {pipeline.provider.name}  (\\q to quit, \\s to toggle sources)", CYAN))
    show_sources = False
    while True:
        try:
            question = input(_c("?> ", CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question in {"\\q", "/q", "exit", "quit"}:
            break
        if question in {"\\s", "/s"}:
            show_sources = not show_sources
            print(_c(f"sources: {'on' if show_sources else 'off'}", DIM))
            continue
        answer = pipeline.ask(question, k=args.k)
        print(answer.text)
        if show_sources:
            _print_sources(answer)
        print()
    return 0


def cmd_eval(args) -> int:
    pipeline = _open_pipeline(args)
    if len(pipeline.store) == 0:
        print(_c("index is empty — ingest a corpus first", DIM))
        return 1
    golden = load_golden(args.golden)
    report = evaluate(pipeline, golden, k=args.k)
    print(report.table())
    return 0


def cmd_info(args) -> int:
    pipeline = _open_pipeline(args)
    stats = pipeline.store.stats()
    print(f"nanorag v{__version__}")
    print(f"  provider : {pipeline.provider.name}")
    print(f"  index    : {pipeline.index_path}")
    print(f"  documents: {stats['documents']}")
    print(f"  chunks   : {stats['chunks']}")
    print(f"  dim      : {stats['dim']}")
    return 0


# ---------------------------------------------------------------------- #
# parser
# ---------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanorag",
        description="Retrieval-Augmented Generation from scratch — no frameworks, no required API keys.",
    )
    parser.add_argument("--version", action="version", version=f"nanorag {__version__}")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="index directory (default: %(default)s)")
    parser.add_argument("--provider", choices=["extractive", "openai", "anthropic", "ollama"], default=None,
                        help="answer provider (default: auto-detect from env, else extractive)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="chunk, embed and index documents")
    p.add_argument("path", help="file or directory to ingest")
    p.add_argument("--patterns", nargs="+", default=["*.md", "*.txt"], help="glob patterns when ingesting a directory")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5, help="passages to retrieve (default: %(default)s)")
    p.add_argument("-s", "--show-sources", action="store_true", help="print retrieved passages")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive REPL")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("eval", help="run the golden-set evaluation")
    p.add_argument("--golden", default="evals/golden.jsonl", help="golden JSONL path (default: %(default)s)")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("info", help="show index statistics")
    p.set_defaults(func=cmd_info)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
