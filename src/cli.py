from __future__ import annotations

import argparse

from src.services.diagnostics_service import DiagnosticsService


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI интерфейс MedDiagnost")
    parser.add_argument("text", nargs="?", help="Запрос пользователя")
    parser.add_argument("--recommend", action="store_true", help="Включить рекомендации")
    args = parser.parse_args()

    service = DiagnosticsService()

    if not args.text:
        print("Введите текст запроса. Пример: python -m src.cli 'кашель и температура'")
        return

    print(service.analyze_text(args.text)["answer"])

    if args.recommend:
        rec = service.recommend_diseases(args.text)
        print("\nРекомендации (cosine):")
        for row in rec["cosine"]:
            print(f"- {row['item']}: {row['score']:.2f}")


if __name__ == "__main__":
    main()
