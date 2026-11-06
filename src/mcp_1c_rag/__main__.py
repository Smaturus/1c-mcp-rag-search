"""Allow: python -m mcp_1c_rag (starts MCP stdio server)."""

from __future__ import annotations

import sys

from mcp_1c_rag.collections import UnknownCollectionError


def _exit_actionable(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def _format_unknown_collection(exc: UnknownCollectionError) -> str:
    return (
        "Неверное имя коллекции в переменной окружения.\n"
        f"{exc}\n"
        "Проверьте MCP_1C_CODE_COLLECTION / MCP_1C_METADATA_COLLECTION.\n"
        "Пример: MCP_1C_CODE_COLLECTION=code_proc "
        "MCP_1C_METADATA_COLLECTION=metadata_xml"
    )


if __name__ == "__main__":
    try:
        from mcp_1c_rag.server import main
        from mcp_1c_rag.startup import RuntimeValidationError, format_validation_error
    except UnknownCollectionError as exc:
        _exit_actionable(_format_unknown_collection(exc))
    except FileNotFoundError as exc:
        _exit_actionable(
            f"Файл не найден: {exc}\n"
            "Проверьте MCP_1C_CHROMA_DIR / MCP_1C_BM25_DIR и сборку индекса."
        )

    try:
        main()
    except RuntimeValidationError as exc:
        _exit_actionable(format_validation_error(exc))
