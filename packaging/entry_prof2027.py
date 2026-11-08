"""Точка входа сборки: запускает CLI PROF_2027.

Используется PyInstaller --onedir: exe рядом с папками models/, config/, data/.
"""

import sys

from prof2027.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
