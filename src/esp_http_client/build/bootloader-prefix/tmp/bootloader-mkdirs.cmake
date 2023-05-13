# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "C:/Users/javga/esp/esp-idf/components/bootloader/subproject"
  "C:/Users/javga/OneDrive/Documentos/master/TFM/esp32-s3/esp_http_client/build/bootloader"
  "C:/Users/javga/OneDrive/Documentos/master/TFM/esp32-s3/esp_http_client/build/bootloader-prefix"
  "C:/Users/javga/OneDrive/Documentos/master/TFM/esp32-s3/esp_http_client/build/bootloader-prefix/tmp"
  "C:/Users/javga/OneDrive/Documentos/master/TFM/esp32-s3/esp_http_client/build/bootloader-prefix/src/bootloader-stamp"
  "C:/Users/javga/OneDrive/Documentos/master/TFM/esp32-s3/esp_http_client/build/bootloader-prefix/src"
  "C:/Users/javga/OneDrive/Documentos/master/TFM/esp32-s3/esp_http_client/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/javga/OneDrive/Documentos/master/TFM/esp32-s3/esp_http_client/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
