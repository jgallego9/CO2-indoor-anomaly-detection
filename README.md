# Detección eficiente de anomalías en sensores de CO2 en tiempo real

## Descripción del proyecto

El seguimiento de la concentración de dióxido de carbono (CO2) es esencial para garantizar una buena calidad del aire en interiores. Sin embargo, la detección de anomalías en los sensores de CO2 puede ser difícil debido a las variaciones en los patrones de los datos y a los límites en la capacidad de procesamiento de los dispositivos. En este trabajo se presenta un sistema de jerarquía de modelos de Machine Learning que utiliza la computación Edge y Cloud para detectar anomalías en los sensores de CO2. El sistema se compone de un modelo más limitado que se ejecuta en el dispositivo (Edge) y de un modelo más complejo que se ejecuta en el Cloud. La selección del modo de trabajo se realiza desde el servidor, lo que permite una detección eficiente y precisa de las anomalías en los sensores de CO2. Los resultados muestran que este sistema proporciona una solución escalable y efectiva para la detección de anomalías en sensores de CO2 en tiempo real, lo que puede mejorar significativamente la calidad del aire en interiores.

Este proyecto forma parte del *Trabajo Fin de Máster* del **Máster Universitario de Ciencia de Datos** de la Universitat Oberta de Catalunya (**UOC**). 
Para una revisión completa del proyecto se puede acudir a la [memoria](TFM_JavierGallego.pdf).

## Conclusiones

Se han probado diversos modelos utilizando técnicas de aprendizaje **semi-supervisado** y **no supervisado** alcanzando buenos resultados. Los modelos desarrollados son **LSTM-Autoencoder**, **Isolation Forest** y **OneClass-SVM**.
Los dos modelos **LSTM-Autoencoder**, tanto el alojado en el servidor como el que se introduce en el microcontrolador, han sido los que han logrado mejores métricas.

Comparativa de modelos:

| **Modelo**           | **TP** | **TN** | **FP** | **FN** | **Accuracy** | **Precision** | **Recall** | **F1-score** |
|----------------------|:------:|:------:|:------:|:------:|:------------:|:-------------:|:----------:|:------------:|
| LSTM-AE (Cloud)      |  1625  |  23797 |   147  |   117  |     0.99     |     0.917     |    0.933   |     0.925    |
| LSTM-AE (Standalone) |   254  |  3995  |   32   |    0   |     0.993    |     0.888     |      1     |     0.941    |
| Isolation Forest     |  1506  |  23793 |   266  |   123  |     0.985    |      0.85     |    0.924   |     0.886    |
| OC-SVM               |  1088  |  23879 |   684  |   37   |     0.972    |     0.614     |    0.967   |     0.751    |

<img width="765" alt="ROC_AUC" src="https://github.com/jgallego9/CO2-indoor-anomaly-detection/assets/38666733/bccf16fc-a2a4-4023-950f-54ecdcbd9ca4">

## Estructura del proyecto

 * [LICENSE](./LICENSE) Licencia del código
 * [README.md](./README.md) Descripción general del repositorio
 * [src](./src) Código Fuente
    * [jupyter](./src/jupyter) Jupyter Notebook en la que se ha realizado la preparación de los datos y los modelos
      * [models.ipynb](./src/jupyter/models.ipynb)
      * [models.html](./src/jupyter/models.html)
    * [anomaly_detection_app](./src/anomaly_detection_app) Aplicación en C++ embebida en la placa de desarrollo del microcontrolador
      * [README.md](./src/anomaly_detection_app/README.md)
      * [sdkconfig.ci](./src/anomaly_detection_app/sdkconfig.ci)
      * [sdkconfig.ci.ssldyn](./src/anomaly_detection_app/sdkconfig.ci.ssldyn)
      * [dependencies.lock](./src/anomaly_detection_app/dependencies.lock)
      * [partition_table.csv](./src/anomaly_detection_app/partition_table.csv)
      * [sdkconfig.old](./src/anomaly_detection_app/sdkconfig.old)
      * [sdkconfig](./src/anomaly_detection_app/sdkconfig)
      * [CMakeLists.txt](./src/anomaly_detection_app/CMakeLists.txt)
      * [components](./src/anomaly_detection_app/components)
        * [bus](./src/anomaly_detection_app/components/bus)
        * [esp-nn](./src/anomaly_detection_app/components/esp-nn)
        * [esp32-camera](./src/anomaly_detection_app/components/esp32-camera)
        * [fb_gfx](./src/anomaly_detection_app/components/fb_gfx)
        * [screen](./src/anomaly_detection_app/components/screen)
        * [tflite-lib](./src/anomaly_detection_app/components/tflite-lib)
      * [main](./src/anomaly_detection_app/main)
        * [input_data.h](./src/anomaly_detection_app/main/input_data.h)
        * [input_norm_data.h](./src/anomaly_detection_app/main/input_norm_data.h)
        * [model.h](./src/anomaly_detection_app/main/model.h)
        * [model.cc](./src/anomaly_detection_app/main/model.cc)
        * [main.cc](./src/anomaly_detection_app/main/main.cc)
        * [CMakeLists.txt](./src/anomaly_detection_app/main/CMakeLists.txt)
      * [Makefile](./src/anomaly_detection_app/Makefile)
    * [server](./src/server) Código en python del servidor
        * [README.md](./src/server/README.md)
        * [static](./src/server/static)
          * [scaler.save](./src/server/static/scaler.save)
          * [LSTM_model.json](./src/server/static/LSTM_model.json)
          * [LSTM_model.h5](./src/server/static/LSTM_model.h5)
          * [index.html](./src/server/static/index.html)
        * [main.py](./src/server/main.py)
 * [TFM_JavierGallego.pdf](./TFM_JavierGallego.pdf) Memoria del TFM

Una descripción más exhaustiva de cada componente se incluye en los ficheros README de cada parte del proyecto.

