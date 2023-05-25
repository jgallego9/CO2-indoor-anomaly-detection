# Cliente HTTP de detección de anomalías de CO2 del microcontrolador 

## Descripción

El cliente HTTP se conecta al servidor mediante Wi-Fi para mandar los datos de los sensores o la inferencia realizada sobre ellos. Al encenderse, el MCU pregunta al servidor en qué modo debe trabajar. Dependiendo de esto se realizará la inferencia en el propio MCU o en el servidor.

Se incorpora un fichero con datos de sensores reales para simular las lecturas de CO2. Si se dispone de un sensor se debe adaptar el código fuente para recibir los datos del sensor en lugar de leerlos de fichero.

El código se ha desarrollado en C++ para la placa de desarrollo de Espressif [ESP32-S3-DevKitC-1](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/hw-reference/esp32s3/user-guide-devkitc-1.html). En concreto la placa ESP32-S3-DevKitC-1-N32R8V.

## Pasos previos

Para el desarrollo y compilación del proyecto se ha utilizado el IDE Visual Studio Code y se ha seguido el [tutorial](https://github.com/espressif/vscode-esp-idf-extension/blob/master/docs/tutorial/install.md) proporcionado por Espressif para usar el framework de desarrollo [esp-idf](https://github.com/espressif/esp-idf).

Una vez tenemos el entorno se debe abrir la configuración del proyecto `ESP-IDF: SDK Configuration editor` e introducir los datos de la red Wi-Fi.

Tras esto se podrá realizar la compilación `ESP-IDF: Build your project` y flashearla en la placa `ESP-IDF: Flash your project`.

## Frameworks utilizados

Para la implementación se han utilizado los [ejemplos proporcionados por Espressif](https://github.com/espressif/tflite-micro-esp-examples) para integrar el port de [Tensorflow Lite para microcontroladores](https://github.com/tensorflow/tflite-micro).
