@echo off
chcp 65001 > nul
title Unity Studio Setup Helper
cd /d "%~dp0"

echo.
echo  ==================================================
echo    AI VTuber Unity Studio Setup Helper
echo  ==================================================
echo.
echo  このバッチファイルは、あなたが指定したUnityプロジェクトに
echo  必要なC#スクリプトと通信用パッケージの設定を自動でインポートします。
echo.

python backend\studio_unity\setup_unity_project.py

pause
