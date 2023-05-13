#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "protocol_examples_common.h"
#include "esp_tls.h"
#if CONFIG_MBEDTLS_CERTIFICATE_BUNDLE
#include "esp_crt_bundle.h"
#endif

#include "esp_http_client.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

#include "input_data.h"
#include "input_norm_data.h"

#define MAX_HTTP_RECV_BUFFER 512
#define MAX_HTTP_OUTPUT_BUFFER 2048
static const char *TAG = "HTTP_CLIENT";

extern const char howsmyssl_com_root_cert_pem_start[] asm("_binary_howsmyssl_com_root_cert_pem_start");
extern const char howsmyssl_com_root_cert_pem_end[]   asm("_binary_howsmyssl_com_root_cert_pem_end");

extern const char postman_root_cert_pem_start[] asm("_binary_postman_root_cert_pem_start");
extern const char postman_root_cert_pem_end[]   asm("_binary_postman_root_cert_pem_end");

esp_http_client_config_t config;

esp_http_client_handle_t client;

char local_response_buffer[MAX_HTTP_OUTPUT_BUFFER] = {0};

int anomaly;

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int scratchBufSize = 39 * 1024;
constexpr int kTensorArenaSize = 81 * 1024 + scratchBufSize;
static uint8_t *tensor_arena;//[kTensorArenaSize];
}

const float THRESHOLD = 6.5301166;

//Una cola para crear la lista de valores de entrada del modelo

#define MAX_QUEUE_SIZE 10

typedef struct {
    float elements[MAX_QUEUE_SIZE];
    int start;
    int end;
} Queue;

void queue_init(Queue *q) {
    q->start = 0;
    q->end = 0;
}

void queue_push(Queue *q, float element) {
    if ((q->end + 1) % MAX_QUEUE_SIZE == q->start) {
        // La cola está llena
        return;
    }
    q->elements[q->end] = element;
    q->end = (q->end + 1) % MAX_QUEUE_SIZE;
}

float queue_pop(Queue *q) {
    if (q->start == q->end) {
        // La cola está vacía
        return 0.0;
    }
    float element = q->elements[q->start];
    q->start = (q->start + 1) % MAX_QUEUE_SIZE;
    return element;
}

int queue_size(Queue *q) {
    if (q->start <= q->end) {
        return q->end - q->start;
    } else {
        return MAX_QUEUE_SIZE - q->start + q->end;
    }
}

Queue input_model;

void setup() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(model_autoencoder_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG,"Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  ESP_LOGI(TAG, "Model setup done!");

  if (tensor_arena == NULL) {
    tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    ESP_LOGE(TAG, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

void predict() {

    TfLiteStatus invoke_status = interpreter->Invoke();

    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return;
    }
}

esp_err_t _http_event_handler(esp_http_client_event_t *evt)
{
    static char *output_buffer;  // Buffer to store response of http request from event handler
    static int output_len;       // Stores number of bytes read
    switch(evt->event_id) {
        case HTTP_EVENT_ERROR:
            ESP_LOGD(TAG, "HTTP_EVENT_ERROR");
            break;
        case HTTP_EVENT_ON_CONNECTED:
            ESP_LOGD(TAG, "HTTP_EVENT_ON_CONNECTED");
            break;
        case HTTP_EVENT_HEADER_SENT:
            ESP_LOGD(TAG, "HTTP_EVENT_HEADER_SENT");
            break;
        case HTTP_EVENT_ON_HEADER:
            ESP_LOGD(TAG, "HTTP_EVENT_ON_HEADER, key=%s, value=%s", evt->header_key, evt->header_value);
            break;
        case HTTP_EVENT_ON_DATA:
            ESP_LOGD(TAG, "HTTP_EVENT_ON_DATA, len=%d", evt->data_len);
            /*
             *  Check for chunked encoding is added as the URL for chunked encoding used in this example returns binary data.
             *  However, event handler can also be used in case chunked encoding is used.
             */
            if (!esp_http_client_is_chunked_response(evt->client)) {
                // If user_data buffer is configured, copy the response into the buffer
                if (evt->user_data) {
                    memcpy(evt->user_data + output_len, evt->data, evt->data_len);
                } else {
                    if (output_buffer == NULL) {
                        output_buffer = (char *) malloc(esp_http_client_get_content_length(evt->client));
                        output_len = 0;
                        if (output_buffer == NULL) {
                            ESP_LOGE(TAG, "Failed to allocate memory for output buffer");
                            return ESP_FAIL;
                        }
                    }
                    memcpy(output_buffer + output_len, evt->data, evt->data_len);
                }
                output_len += evt->data_len;
            }

            break;
        case HTTP_EVENT_ON_FINISH:
            ESP_LOGD(TAG, "HTTP_EVENT_ON_FINISH");
            if (output_buffer != NULL) {
                // Response is accumulated in output_buffer. Uncomment the below line to print the accumulated response
                // ESP_LOG_BUFFER_HEX(TAG, output_buffer, output_len);
                free(output_buffer);
                output_buffer = NULL;
            }
            output_len = 0;
            break;
        case HTTP_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "HTTP_EVENT_DISCONNECTED");
            int mbedtls_err = 0;
            esp_err_t err = esp_tls_get_and_clear_last_error((esp_tls_error_handle_t)evt->data, &mbedtls_err, NULL);
            if (err != 0) {
                ESP_LOGI(TAG, "Last esp error code: 0x%x", err);
                ESP_LOGI(TAG, "Last mbedtls failure: 0x%x", mbedtls_err);
            }
            if (output_buffer != NULL) {
                free(output_buffer);
                output_buffer = NULL;
            }
            output_len = 0;
            break;
    }
    return ESP_OK;
}

static void http_get_mode(void)
{
    // GET

    esp_http_client_set_url(client, "/get_mode/");
    esp_http_client_set_method(client, HTTP_METHOD_GET);
    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "HTTP GET Status = %d, content_length = %d",
                esp_http_client_get_status_code(client),
                esp_http_client_get_content_length(client));
    } else {
        ESP_LOGE(TAG, "HTTP GET request failed: %s", esp_err_to_name(err));
    }

}

static void http_post_sensor_data(float input_value)
{
    // POST
    char post_data[50];
    sprintf(post_data, "{\"value\": %.1f}", input_value);

    esp_http_client_set_url(client, "/predict/");
    esp_http_client_set_method(client, HTTP_METHOD_POST);
    esp_http_client_set_header(client, "Content-Type", "application/json");
    esp_http_client_set_post_field(client, post_data, strlen(post_data));
    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "HTTP POST Status = %d, content_length = %d",
                esp_http_client_get_status_code(client),
                esp_http_client_get_content_length(client));
    } else {
        ESP_LOGE(TAG, "HTTP POST request failed: %s", esp_err_to_name(err));
    }

}

static void http_post_prediction(float input_value, int32_t anomaly)
{
    // POST
    char post_data[50];
    sprintf(post_data, "{\"value\": %.1f, \"anomaly\": %d}", input_value, anomaly);

    esp_http_client_set_url(client, "/anomaly/");
    esp_http_client_set_method(client, HTTP_METHOD_POST);
    esp_http_client_set_header(client, "Content-Type", "application/json");
    esp_http_client_set_post_field(client, post_data, strlen(post_data));
    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "HTTP POST Status = %d, content_length = %d",
                esp_http_client_get_status_code(client),
                esp_http_client_get_content_length(client));
    } else {
        ESP_LOGE(TAG, "HTTP POST request failed: %s", esp_err_to_name(err));
    }

}

static void http_post_data(void *pvParameters)
{
    http_get_mode();
    ESP_LOGI(TAG, "Mode: %s", local_response_buffer);

    config = (esp_http_client_config_t){
        .host = "192.168.1.131",
        .port = 8000,
        .path = "/",
        .event_handler = _http_event_handler,
        .transport_type = HTTP_TRANSPORT_OVER_TCP,
    };
    client = esp_http_client_init(&config);

    sleep(10);
    if (strcmp(local_response_buffer, "\"cloud\"") == 0) {
        ESP_LOGI(TAG, "Starting Cloud execution...");
        sleep(5);
        int i;
        int input_len = sizeof(input_data) / sizeof(input_data[0]);
        for (i = 0; i < input_len; i++) {
            ESP_LOGI(TAG, "Data = %f,", input_data[i]);
            http_post_sensor_data(input_data[i]);
            sleep(1);
        }
    }
    else {
        ESP_LOGI(TAG, "Starting Standalone execution...");
        sleep(5);
        int i;
        int size;
        int input_len = sizeof(input_norm_data) / sizeof(input_norm_data[0]);
        for (i = 0; i < input_len; i++) {
            //float dist = 0.0;
            //float feature[] = { input_norm_data[i] };
            //int32_t predicted_class = elliptic_envelope_predict(feature, 1);
            //int32_t predicted_class = eml_elliptic_envelope_predict(&elliptic_envelope_classifier, feature, 1, &dist);
            //ESP_LOGI(TAG, "Prediction = %d,", predicted_class);
            queue_push(&input_model, input_norm_data[i]);
            size = queue_size(&input_model);
            if (size == MAX_QUEUE_SIZE - 1) {
                ESP_LOGI(TAG, "Data = %f,", input_data[i]);
                for (int j = 0; j < MAX_QUEUE_SIZE; j++) {
                    input->data.f[j] = input_model.elements[j];
                }
                predict();
                if (output->data.f[MAX_QUEUE_SIZE-1] > THRESHOLD) {
                    anomaly = 1;
                }
                else {
                    anomaly = 0;
                }
                ESP_LOGI(TAG, "Prediction = %d,", anomaly);
                http_post_prediction(input_data[i], anomaly);
                queue_pop(&input_model);
                sleep(1);
            }
        }
    }
    esp_http_client_cleanup(client);
    ESP_LOGI(TAG, "Finish http connection");
    vTaskDelete(NULL);
}

extern "C" void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    /* This helper function configures Wi-Fi or Ethernet, as selected in menuconfig.
     * Read "Establishing Wi-Fi or Ethernet Connection" section in
     * examples/protocols/README.md for more information about this function.
     */
    ESP_ERROR_CHECK(example_connect());
    ESP_LOGI(TAG, "Connected to AP, begin http connection");

    config = (esp_http_client_config_t){
        .host = "192.168.1.131",
        .port = 8000,
        .path = "/",
        .event_handler = _http_event_handler,
        .transport_type = HTTP_TRANSPORT_OVER_TCP,
        .user_data = local_response_buffer,
    };

    client = esp_http_client_init(&config);
    setup();
    queue_init(&input_model);
    
    sleep(10);
    ESP_LOGI(TAG, "Begin sending data");
    xTaskCreate(&http_post_data, "http_post_data", 8192, NULL, 5, NULL);

}