@echo off
setlocal
cd /d "%~dp0"
py -3 --version >nul 2>&1
if errorlevel 1 goto try_python
set "LOCAL_RAG_PY=py -3"
goto run
:try_python
python --version >nul 2>&1
if errorlevel 1 goto missing
set "LOCAL_RAG_PY=python"
:run
echo Downloading the complete IBM corpus and building local search indexes.
echo This needs internet and may take several minutes.
%LOCAL_RAG_PY% -B annotate_server\corpus_store.py --download
pause
exit /b
:missing
echo Python 3.10 or newer is required. Install Python and run this file again.
pause
exit /b 1
