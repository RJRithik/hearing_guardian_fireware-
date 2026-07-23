/*
  AI-Powered Industrial Hearing Guardian - Firmware (v3)
  ESP32 + INMP441 (I2S Mic) + Edge Impulse Model + Vibration Motor + LED Alert

  Built on Edge Impulse's official esp32_microphone_continuous example,
  using YOUR actual exported model (Industrial-Hearing-Guardian_inferencing).

  ============ WIRING (matches this code) ============
  INMP441 VDD  -> ESP32 3.3V
  INMP441 GND  -> ESP32 GND
  INMP441 L/R  -> ESP32 GND   (sets channel to Left)
  INMP441 SCK  -> ESP32 GPIO 26
  INMP441 WS   -> ESP32 GPIO 32
  INMP441 SD   -> ESP32 GPIO 33

  Vibration Motor (via transistor switch) -> ESP32 GPIO 27
  Red Alert LED (with resistor)           -> ESP32 GPIO 25
  =====================================================

  CHANGES IN v3 (addressing specific review feedback):
  1. I2S_PORT now uses the named macro I2S_NUM_0 everywhere instead of the
     magic number (i2s_port_t)1 - improves portability across ESP32 variants.
  2. Hysteresis upgraded from "N strict consecutive hits" to a sliding
     window (2-of-last-4 slices) - survives a single borderline slice
     (e.g. 84% instead of 85%) without resetting progress to zero.
  3. The blocking wait in microphone_inference_record() now has a timeout.
     If the I2S/DMA task stalls or crashes, the device logs the fault and
     performs a controlled esp_restart() instead of hanging forever.
  4. I2S driver calls are now version-guarded: uses the modern i2s_std
     channel API on newer ESP-IDF/Arduino core (IDF 5.x+), and falls back
     to the legacy driver on older cores - so it keeps working today and
     won't silently rot when you update your board package later.
*/

#define EIDSP_QUANTIZE_FILTERBANK   0

#include <Industrial-Hearing-Guardian_inferencing.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_idf_version.h"

#if ESP_IDF_VERSION_MAJOR >= 5
  #define USE_NEW_I2S_DRIVER 1
  #include "driver/i2s_std.h"
#else
  #define USE_NEW_I2S_DRIVER 0
  #include "driver/i2s.h"
#endif

// ================= I2S PORT (named macro, not a magic number) =================
#define I2S_PORT      I2S_NUM_0

// ================= ALERT OUTPUT PINS =================
#define MOTOR_PIN    27
#define LED_PIN      25

// ================= ALERT SETTINGS =================
#define CONFIDENCE_THRESHOLD     0.85f   // 85% as per your project spec
#define ALERT_DURATION_MS        1500
#define ALERT_COOLDOWN_MS        2000    // minimum gap between separate alerts

// ---- Sliding-window hysteresis: trigger if HITS_NEEDED of last WINDOW_SIZE slices are hazards ----
#define HAZARD_WINDOW_SIZE       4
#define HAZARD_HITS_NEEDED       2

// ================= AUDIO GAIN =================
#define AUDIO_GAIN               8

// ================= RELIABILITY / WATCHDOG =================
#define RECORD_TIMEOUT_MS        3000    // if no audio buffer arrives in this window, treat as a stall

/** Audio buffers, pointers and selectors */
typedef struct {
    signed short *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static const uint32_t sample_buffer_size = 2048;
static signed short sampleBuffer[sample_buffer_size];
static bool debug_nn = false;
static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);
static bool record_status = true;

#if USE_NEW_I2S_DRIVER
static i2s_chan_handle_t rx_handle = NULL;
#endif

// ---- Non-blocking alert state ----
static bool alert_active = false;
static unsigned long alert_start_ms = 0;
static unsigned long last_alert_end_ms = 0;

// ---- Sliding-window hazard tracking ----
static bool hazard_history[HAZARD_WINDOW_SIZE] = {false};
static int hazard_history_index = 0;

// ---- Diagnostics ----
static unsigned long last_heap_log_ms = 0;

void start_alert(const char* label) {
    unsigned long now = millis();

    if (now - last_alert_end_ms < ALERT_COOLDOWN_MS) {
        return;
    }

    Serial.print(">>> ALERT TRIGGERED: ");
    Serial.println(label);
    digitalWrite(MOTOR_PIN, HIGH);
    digitalWrite(LED_PIN, HIGH);
    alert_active = true;
    alert_start_ms = now;
}

void update_alert_state() {
    if (alert_active && (millis() - alert_start_ms >= ALERT_DURATION_MS)) {
        digitalWrite(MOTOR_PIN, LOW);
        digitalWrite(LED_PIN, LOW);
        alert_active = false;
        last_alert_end_ms = millis();
    }
}

bool record_hazard_slice(bool is_hazard) {
    hazard_history[hazard_history_index] = is_hazard;
    hazard_history_index = (hazard_history_index + 1) % HAZARD_WINDOW_SIZE;

    int hit_count = 0;
    for (int i = 0; i < HAZARD_WINDOW_SIZE; i++) {
        if (hazard_history[i]) hit_count++;
    }
    return hit_count >= HAZARD_HITS_NEEDED;
}

void clear_hazard_history() {
    for (int i = 0; i < HAZARD_WINDOW_SIZE; i++) {
        hazard_history[i] = false;
    }
}

void setup()
{
    Serial.begin(115200);
    while (!Serial);
    Serial.println("Hearing Guardian v3 - Edge Impulse Inferencing Demo");
    Serial.printf("I2S driver mode: %s\n", USE_NEW_I2S_DRIVER ? "modern i2s_std" : "legacy i2s");

    pinMode(MOTOR_PIN, OUTPUT);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(MOTOR_PIN, LOW);
    digitalWrite(LED_PIN, LOW);

    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: ");
    ei_printf_float((float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf(" ms.\n");
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

    run_classifier_init();
    ei_printf("\nStarting continuous inference in 2 seconds...\n");
    ei_sleep(2000);

    if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
        ei_printf("ERR: Could not allocate audio buffer (size %d), this could be due to the window length of your model\r\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
        return;
    }

    ei_printf("Recording...\n");
    Serial.printf("Free heap at start: %d bytes\n", ESP.getFreeHeap());
}

void loop()
{
    update_alert_state();

    bool m = microphone_inference_record();
    if (!m) {
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    bool hazard_this_slice = false;
    const char* hazard_label = "";

    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        const char* label = result.classification[ix].label;
        float confidence = result.classification[ix].value;

        bool is_hazard = (strcmp(label, "scream") == 0) || (strcmp(label, "siren") == 0);

        if (is_hazard && confidence > CONFIDENCE_THRESHOLD) {
            hazard_this_slice = true;
            hazard_label = label;
        }
    }

    bool should_fire = record_hazard_slice(hazard_this_slice);
    if (should_fire) {
        start_alert(hazard_label[0] ? hazard_label : "hazard");
        clear_hazard_history();
    }

    if (++print_results >= (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)) {
        ei_printf("Predictions ");
        ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);
        ei_printf(": \n");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("    %s: ", result.classification[ix].label);
            ei_printf_float(result.classification[ix].value);
            ei_printf("\n");
        }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
        ei_printf("    anomaly score: ");
        ei_printf_float(result.anomaly);
        ei_printf("\n");
#endif
        print_results = 0;
    }

    if (millis() - last_heap_log_ms > 10000) {
        Serial.printf("[diag] Free heap: %d bytes\n", ESP.getFreeHeap());
        last_heap_log_ms = millis();
    }
}

static void audio_inference_callback(uint32_t n_bytes)
{
    for(int i = 0; i < n_bytes>>1; i++) {
        inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];

        if(inference.buf_count >= inference.n_samples) {
            inference.buf_select ^= 1;
            inference.buf_count = 0;
            inference.buf_ready = 1;
        }
    }
}

static void capture_samples(void* arg) {
  const int32_t i2s_bytes_to_read = (uint32_t)arg;
  size_t bytes_read = i2s_bytes_to_read;

  while (record_status) {

#if USE_NEW_I2S_DRIVER
    esp_err_t read_err = i2s_channel_read(rx_handle, (void*)sampleBuffer, i2s_bytes_to_read, &bytes_read, portMAX_DELAY);
    if (read_err != ESP_OK) {
        ei_printf("Error in I2S channel read : %d", read_err);
        vTaskDelay(pdMS_TO_TICKS(10));
        continue;
    }
#else
    i2s_read(I2S_PORT, (void*)sampleBuffer, i2s_bytes_to_read, &bytes_read, 100);
#endif

    if (bytes_read <= 0) {
      ei_printf("Error in I2S read : %d", bytes_read);
    }
    else {
        if (bytes_read < i2s_bytes_to_read) {
            ei_printf("Partial I2S read - clearing buffer to avoid stale data\n");
            memset((uint8_t*)sampleBuffer + bytes_read, 0, i2s_bytes_to_read - bytes_read);
        }

        for (int x = 0; x < i2s_bytes_to_read/2; x++) {
            int32_t scaled = (int32_t)sampleBuffer[x] * AUDIO_GAIN;
            if (scaled > INT16_MAX) scaled = INT16_MAX;
            if (scaled < INT16_MIN) scaled = INT16_MIN;
            sampleBuffer[x] = (int16_t)scaled;
        }

        if (record_status) {
            audio_inference_callback(i2s_bytes_to_read);
        }
        else {
            break;
        }
    }
  }
  vTaskDelete(NULL);
}

static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));
    if (inference.buffers[0] == NULL) {
        return false;
    }
    inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));
    if (inference.buffers[1] == NULL) {
        ei_free(inference.buffers[0]);
        inference.buffers[0] = NULL;
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    if (i2s_init(EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start I2S!");
        ei_free(inference.buffers[0]);
        ei_free(inference.buffers[1]);
        inference.buffers[0] = NULL;
        inference.buffers[1] = NULL;
        return false;
    }
    ei_sleep(100);
    record_status = true;
    xTaskCreate(capture_samples, "CaptureSamples", 1024 * 32, (void*)sample_buffer_size, 10, NULL);
    return true;
}

static bool microphone_inference_record(void)
{
    if (inference.buf_ready == 1) {
        ei_printf(
            "Error sample buffer overrun. Decrease the number of slices per model window "
            "(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)\n");
        return false;
    }

    unsigned long wait_start = millis();
    while (inference.buf_ready == 0) {
        delay(1);

        if (millis() - wait_start > RECORD_TIMEOUT_MS) {
            Serial.println("FATAL: Audio capture stalled (no data for RECORD_TIMEOUT_MS). "
                            "Restarting device to recover I2S/DMA state...");
            delay(50);
            esp_restart();
            return false;
        }
    }
    inference.buf_ready = 0;
    return true;
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);
    return 0;
}

static void microphone_inference_end(void)
{
    i2s_deinit();
    if (inference.buffers[0]) ei_free(inference.buffers[0]);
    if (inference.buffers[1]) ei_free(inference.buffers[1]);
    inference.buffers[0] = NULL;
    inference.buffers[1] = NULL;
}

static int i2s_init(uint32_t sampling_rate) {

#if USE_NEW_I2S_DRIVER
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_PORT, I2S_ROLE_MASTER);
    esp_err_t ret = i2s_new_channel(&chan_cfg, NULL, &rx_handle);
    if (ret != ESP_OK) {
        ei_printf("Error in i2s_new_channel");
        return int(ret);
    }

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(sampling_rate),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = (gpio_num_t)26,
            .ws   = (gpio_num_t)32,
            .dout = I2S_GPIO_UNUSED,
            .din  = (gpio_num_t)33,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ret = i2s_channel_init_std_mode(rx_handle, &std_cfg);
    if (ret != ESP_OK) {
        ei_printf("Error in i2s_channel_init_std_mode");
        return int(ret);
    }

    ret = i2s_channel_enable(rx_handle);
    if (ret != ESP_OK) {
        ei_printf("Error in i2s_channel_enable");
    }
    return int(ret);

#else
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_TX),
        .sample_rate = sampling_rate,
        .bits_per_sample = (i2s_bits_per_sample_t)16,
        .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = 0,
        .dma_buf_count = 8,
        .dma_buf_len = 512,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = -1,
    };
    i2s_pin_config_t pin_config = {
        .bck_io_num = 26,
        .ws_io_num = 32,
        .data_out_num = -1,
        .data_in_num = 33,
    };
    esp_err_t ret = 0;

    ret = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    if (ret != ESP_OK) {
        ei_printf("Error in i2s_driver_install");
        return int(ret);
    }
    ret = i2s_set_pin(I2S_PORT, &pin_config);
    if (ret != ESP_OK) {
        ei_printf("Error in i2s_set_pin");
        return int(ret);
    }
    ret = i2s_zero_dma_buffer(I2S_PORT);
    if (ret != ESP_OK) {
        ei_printf("Error in initializing dma buffer with 0");
    }
    return int(ret);
#endif
}

static int i2s_deinit(void) {
#if USE_NEW_I2S_DRIVER
    if (rx_handle) {
        i2s_channel_disable(rx_handle);
        i2s_del_channel(rx_handle);
        rx_handle = NULL;
    }
#else
    i2s_driver_uninstall(I2S_PORT);
#endif
    return 0;
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
