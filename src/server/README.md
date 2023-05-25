# Red
netstat -aon
taskkill /pid 20308 /F
netsh advfirewall firewall delete rule name="Allowing LAN connections"
netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=0.0.0.0
uvicorn main:app --host 0.0.0.0
netsh advfirewall firewall add rule name="Allowing LAN connections" dir=in action=allow protocol=TCP localport=8000
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=localhost
## Esperar a que estea dispoñible

# Abrir miniconda y moverse a este directorio
conda activate tfm
conda install keras
pip install "fastapi[all]"
pip install sse-starlette
pip install scikit-learn
pip install joblib
uvicorn main:app --host 0.0.0.0 --reload