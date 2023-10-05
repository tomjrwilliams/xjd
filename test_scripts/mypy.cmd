call ./__local__/setpaths.cmd
set MYPYPATH=./__local__;%xtuples%;%xfactors%;
python -m mypy .%1 --check-untyped-defs --soft-error-limit=-1 | python ./test_scripts/filter_mypy.py