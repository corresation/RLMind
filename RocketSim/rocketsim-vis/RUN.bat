@echo off
rem Opens the visualizer and plays the newest save from any build folder, switching to newer saves while training runs
start "RLMind render" /d "%~dp0..\.." cmd /k python main.py render

python "%~dp0src\main.py"

taskkill /fi "WINDOWTITLE eq RLMind render*" /t /f >nul
