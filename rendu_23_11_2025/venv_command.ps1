python -m venv python_env
python_env\Scripts\Activate.ps1
pip install -r requirements.txt
python.exe -m pip install --upgrade pip

pip freeze > requirements.txt