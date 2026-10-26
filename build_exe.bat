@echo off
echo Construyendo el ejecutable con icono y metadatos...
python -m PyInstaller --noconsole --noconfirm --name "4K_Background_Remover" --icon "app_icon.ico" --add-data "app_icon.ico;." --collect-all customtkinter --collect-all imageio --copy-metadata imageio --copy-metadata imageio_ffmpeg --copy-metadata tqdm --hidden-import torch_directml app_gui.py
echo.
echo Compilacion terminada. El archivo ejecutable esta en la carpeta "dist\4K_Background_Remover".
pause
