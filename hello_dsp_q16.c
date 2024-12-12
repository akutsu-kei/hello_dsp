/**
 * Copyright (c) 2021 Raspberry Pi (Trading) Ltd.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <stdio.h>
#include <arm_math.h>
#include "pico/stdlib.h"
#include "hardware/gpio.h"

#include "image_data.h"
#include "model_parameters.h"

#define FC1_STATE_SIZE (1 + FC1_SIZE) *INPUT_SIZE
#define FC2_STATE_SIZE (1 + FC2_SIZE) *FC1_SIZE

q15_t input_data_q15[INPUT_SIZE];
q15_t fc1_output_q15[FC1_SIZE];
q15_t fc2_output_q15[FC2_SIZE];

void relu_q15(q15_t *data, uint32_t size) {
    for(uint32_t i = 0; i < size; i++) {
        if(data[i] < 0) {
            data[i] = 0;
        }
    }
}

void q31_to_float(q31_t *input, float32_t *output, uint32_t size, float scale, int zero_point) {
    for(uint32_t i = 0; i < size; i++) {
        output[i] = ((float)input[i] / (float)(1 << 31)) * scale + zero_point;
    }
}

arm_status mlp_forward(const q15_t *input, q15_t *fc2_output_q15) {
    arm_status status;

    arm_matrix_instance_q15 mat_input;
    mat_input.numRows = 1;
    mat_input.numCols = INPUT_SIZE;
    mat_input.pData = (q15_t *)input;

    arm_matrix_instance_q15 mat_fc1_weights_instance;
    mat_fc1_weights_instance.numRows = INPUT_SIZE;
    mat_fc1_weights_instance.numCols = FC1_SIZE;
    mat_fc1_weights_instance.pData = (q15_t *)fc1_weight;

    arm_matrix_instance_q15 mat_fc1_output_instance;
    mat_fc1_output_instance.numRows = 1;
    mat_fc1_output_instance.numCols = FC1_SIZE;
    mat_fc1_output_instance.pData = fc1_output_q15;

    q15_t mat_mult_state_fc1[FC1_STATE_SIZE];
    #ifdef FAST_CMSIS_INSTRUCTION
    status = arm_mat_mult_fast_q15(&mat_input, &mat_fc1_weights_instance, &mat_fc1_output_instance, mat_mult_state_fc1);
    #else
    status = arm_mat_mult_q15(&mat_input, &mat_fc1_weights_instance, &mat_fc1_output_instance, mat_mult_state_fc1);
    #endif
    if(status != ARM_MATH_SUCCESS) {
        return status;
    }

    relu_q15(fc1_output_q15, FC1_SIZE);

    arm_matrix_instance_q15 mat_fc2_weights_instance;
    mat_fc2_weights_instance.numRows = FC1_SIZE;
    mat_fc2_weights_instance.numCols = FC2_SIZE;
    mat_fc2_weights_instance.pData = (q15_t *)fc2_weight;

    arm_matrix_instance_q15 mat_fc2_output_instance;
    mat_fc2_output_instance.numRows = 1;
    mat_fc2_output_instance.numCols = FC2_SIZE;
    mat_fc2_output_instance.pData = fc2_output_q15;

    q15_t mat_mult_state_fc2[FC2_STATE_SIZE];
    #ifdef FAST_CMSIS_INSTRUCTION
    status = arm_mat_mult_fast_q15(&mat_fc1_output_instance, &mat_fc2_weights_instance, &mat_fc2_output_instance, mat_mult_state_fc2);
    #else
    status = arm_mat_mult_q15(&mat_fc1_output_instance, &mat_fc2_weights_instance, &mat_fc2_output_instance, mat_mult_state_fc2);
    #endif
    if(status != ARM_MATH_SUCCESS) {
        return status;
    }

    return ARM_MATH_SUCCESS;
}

int pipeline(const q15_t* image, int *index_of_maximum) {
    q15_t output_data[FC2_SIZE];

    arm_status status  = mlp_forward(image, output_data);
    if(status != ARM_MATH_SUCCESS) {
        printf("Error: MLP forward failed.\n");
        return -1;
    }

    q15_t maximum = -32768;
    for(int i = 0; i < FC2_SIZE; i++) {
        if (output_data[i] > maximum) {
            maximum = output_data[i];
            *index_of_maximum = i;
        }
    }
    printf("Index of maximum: %d\n", *index_of_maximum);
    return 0;
}

int main() {
    stdio_init_all();

    const uint LED_PIN = 16;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    while(true) {
        gpio_put(LED_PIN, 1);
        int index_of_maximum;
        pipeline(image_0, &index_of_maximum);
        assert(index_of_maximum == 0);
        pipeline(image_1, &index_of_maximum);
        assert(index_of_maximum == 1);
        pipeline(image_2, &index_of_maximum);
        assert(index_of_maximum == 2);
        pipeline(image_3, &index_of_maximum);
        assert(index_of_maximum == 3);
        pipeline(image_4, &index_of_maximum);
        assert(index_of_maximum == 4);
        pipeline(image_5, &index_of_maximum);
        assert(index_of_maximum == 5);
        pipeline(image_6, &index_of_maximum);
        assert(index_of_maximum == 6);
        pipeline(image_7, &index_of_maximum);
        assert(index_of_maximum == 7);
        pipeline(image_8, &index_of_maximum);
        assert(index_of_maximum == 8);
        pipeline(image_9, &index_of_maximum);
        assert(index_of_maximum == 9);
        gpio_put(LED_PIN, 0);
        sleep_ms(2000);
    }

    return 0;
}
