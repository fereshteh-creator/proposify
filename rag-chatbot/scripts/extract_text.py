import os
import sys
from pathlib import Path

import fitz  # PyMuPDF


def main() -> int:
    # Pfade relativ zur Skriptdatei (nicht CWD) auflösen
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    input_folder = project_root / "data"
    output_folder = input_folder / "text"

    # Ordner erstellen, falls nicht vorhanden
    output_folder.mkdir(parents=True, exist_ok=True)

    # PDF-Dateien ermitteln
    pdf_files = sorted(p for p in input_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"Keine PDFs im Ordner gefunden: {input_folder}")
        return 0

    # Alle PDFs im Input-Ordner durchgehen
    for pdf_path in pdf_files:
        print(f"Bearbeite: {pdf_path.name}")

        # PDF öffnen und Text extrahieren
        doc = fitz.open(str(pdf_path))
        try:
            full_text = ""
            for page_num, page in enumerate(doc, start=1):
                full_text += f"\n--- Seite {page_num} ---\n"
                full_text += page.get_text()
        finally:
            doc.close()

        # Text als .txt speichern
        txt_path = output_folder / f"{pdf_path.stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

    print(f"Alle Texte wurden extrahiert und gespeichert in: {output_folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

