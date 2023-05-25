# Servidor de la aplicación de detección de anomalías de CO2

## Descripción

Se utiliza FastAPI para desplegar un servidor python que presenta una API para recibir datos del microcontrolador y mostrarlos en el cliente web.

## Instalación de librerías necesarias

Para la instalación de paquetes, puedes utilizar un sistema de administración de entornos como Conda o instalarlos directamente con pip.

```
pip install keras 
pip install "fastapi[all]" 
pip install sse-starlette 
pip install scikit-learn 
pip install joblib
```

## Lanzar servidor

Se lanza el servidor con `uvicorn main:app --host 0.0.0.0:8000`.

Para acceder al cliente web se puede acceder por navegador a la dirección http://localhost:8000.

