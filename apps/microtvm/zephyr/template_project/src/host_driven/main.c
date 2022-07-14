/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * This is a sample Zephyr-based application that contains the logic
 * needed to control a microTVM-based model via the UART. This is only
 * intended to be a demonstration, since typically you will want to incorporate
 * this logic into your own application.
 */
#include <drivers/gpio.h>
#include <drivers/uart.h>
#include <fatal.h>
#include <kernel.h>
#include <random/rand32.h>
#include <stdio.h>
#include <sys/printk.h>
#include <sys/reboot.h>
#include <sys/ring_buffer.h>
#include <timing/timing.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/microtvm_rpc_server.h>
#include <unistd.h>
#include <zephyr.h>

#ifdef FVP
#include <irq.h>
#endif

#ifdef CONFIG_ARCH_POSIX
#include "posix_board_if.h"
#endif

#include "crt_config.h"

static const struct device* tvm_uart;

#ifdef CONFIG_LED
#define LED0_NODE DT_ALIAS(led0)
#define LED0 DT_GPIO_LABEL(LED0_NODE, gpios)
#define LED0_PIN DT_GPIO_PIN(LED0_NODE, gpios)
#define LED0_FLAGS DT_GPIO_FLAGS(LED0_NODE, gpios)
static const struct device* led0_pin;
#endif  // CONFIG_LED

static size_t g_num_bytes_requested = 0;
static size_t g_num_bytes_written = 0;
static size_t g_num_bytes_in_rx_buffer = 0;

#ifdef FVP
void uart_log(const char* c) {
  while (*c) {
    uart_poll_out(tvm_uart, *c);
    c++;
  }
}

static uint32_t semihost_cmd(uint32_t opcode, void* arg) {
  uint32_t ret_val;
  __asm__ volatile(
      "mov r0, %[opcode]\n\t"
      "mov r1, %[arg]\n\t"
      "bkpt #0xab\n\r"
      "mov %[ret_val], r0"
      : [ ret_val ] "=r"(ret_val)
      : [ opcode ] "r"(opcode), [ arg ] "r"(arg)
      : "r1", "memory");

  return ret_val;
}

int32_t stdout_fd;
int32_t stdin_fd;

void init_semihosting() {
  // https://github.com/ARM-software/abi-aa/blob/main/semihosting/semihosting.rst#sys-open-0x01
  struct {
    const char* file_name;
    uint32_t mode;
    uint32_t file_name_len;
  } params;
  params.file_name = ":tt";
  params.mode = 5;  // "wb"
  params.file_name_len = 3;
  stdout_fd = semihost_cmd(0x01, &params);
  if (stdout_fd < 0) {
    char err[30];
    snprintf(err, 30, "stdout open: %x", stdout_fd);
    uart_log(err);
  }

  params.mode = 0;
  stdin_fd = semihost_cmd(0x01, &params);
  if (stdin_fd < 0) {
    char err[30];
    snprintf(err, 30, "stdin open: %x", stdin_fd);
    uart_log(err);
  }

  for (int i = 0; i < 200; ++i) {
    // char err[10];
    // snprintf(err, 10, "%d :", i);
    // uart_log(err);
    uart_log("dummy log...\n");
  }
}

ssize_t read_serial(uint8_t* data, size_t size) {
  struct {
    uint32_t file_handle;
    const uint8_t* data;
    uint32_t size;
  } read_req;
  read_req.file_handle = stdin_fd;
  read_req.data = data;
  read_req.size = size;
  uint32_t ret_val = semihost_cmd(0x06, &read_req);
  return size - ret_val;
}
#endif

// Called by TVM to write serial data to the UART.
ssize_t write_serial(void* unused_context, const uint8_t* data, size_t size) {
#ifdef FVP
  struct {
    uint32_t file_handle;
    const uint8_t* data;
    uint32_t size;
  } write_req;
  write_req.file_handle = stdout_fd;
  write_req.data = data;
  write_req.size = size;
  // char msg[30];
  // snprintf(msg, 30, "write %d bytes\n", size);
  // uart_log(msg);
  uint32_t ret_val = semihost_cmd(0x05, &write_req);
  return size - ret_val;
#else
#ifdef CONFIG_LED
  gpio_pin_set(led0_pin, LED0_PIN, 1);
#endif
  g_num_bytes_requested += size;

  for (size_t i = 0; i < size; i++) {
    uart_poll_out(tvm_uart, data[i]);
    g_num_bytes_written++;
  }

#ifdef CONFIG_LED
  gpio_pin_set(led0_pin, LED0_PIN, 0);
#endif

  return size;
#endif
}

// This is invoked by Zephyr from an exception handler, which will be invoked
// if the device crashes. Here, we turn on the LED and spin.
void k_sys_fatal_error_handler(unsigned int reason, const z_arch_esf_t* esf) {
#ifdef CONFIG_LED
  gpio_pin_set(led0_pin, LED0_PIN, 1);
#endif
  for (;;)
    ;
}

// Called by TVM when a message needs to be formatted.
size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  return vsnprintk(out_buf, out_buf_size_bytes, fmt, args);
}

// Called by TVM when an internal invariant is violated, and execution cannot continue.
void TVMPlatformAbort(tvm_crt_error_t error) {
  TVMLogf("TVMError: 0x%x", error);
  sys_reboot(SYS_REBOOT_COLD);
#ifdef CONFIG_LED
  gpio_pin_set(led0_pin, LED0_PIN, 1);
#endif
  for (;;)
    ;
}

// Called by TVM to generate random data.
tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes) {
  uint32_t random;  // one unit of random data.

  // Fill parts of `buffer` which are as large as `random`.
  size_t num_full_blocks = num_bytes / sizeof(random);
  for (int i = 0; i < num_full_blocks; ++i) {
    random = sys_rand32_get();
    memcpy(&buffer[i * sizeof(random)], &random, sizeof(random));
  }

  // Fill any leftover tail which is smaller than `random`.
  size_t num_tail_bytes = num_bytes % sizeof(random);
  if (num_tail_bytes > 0) {
    random = sys_rand32_get();
    memcpy(&buffer[num_bytes - num_tail_bytes], &random, num_tail_bytes);
  }
  return kTvmErrorNoError;
}

// Heap for use by TVMPlatformMemoryAllocate.
K_HEAP_DEFINE(tvm_heap, 216 * 1024);

// Called by TVM to allocate memory.
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  *out_ptr = k_heap_alloc(&tvm_heap, num_bytes, K_NO_WAIT);
  return (*out_ptr == NULL) ? kTvmErrorPlatformNoMemory : kTvmErrorNoError;
}

// Called by TVM to deallocate memory.
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  k_heap_free(&tvm_heap, ptr);
  return kTvmErrorNoError;
}

volatile timing_t g_microtvm_start_time, g_microtvm_end_time;
int g_microtvm_timer_running = 0;

// Called to start system timer.
tvm_crt_error_t TVMPlatformTimerStart() {
  if (g_microtvm_timer_running) {
    TVMLogf("timer already running");
    return kTvmErrorPlatformTimerBadState;
  }

#ifdef CONFIG_LED
  gpio_pin_set(led0_pin, LED0_PIN, 1);
#endif
  g_microtvm_start_time = timing_counter_get();
  g_microtvm_timer_running = 1;
  return kTvmErrorNoError;
}

// Called to stop system timer.
tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  if (!g_microtvm_timer_running) {
    TVMLogf("timer not running");
    return kTvmErrorSystemErrorMask | 2;
  }

#ifdef CONFIG_LED
  gpio_pin_set(led0_pin, LED0_PIN, 0);
#endif

  g_microtvm_end_time = timing_counter_get();
  uint64_t cycles = timing_cycles_get(&g_microtvm_start_time, &g_microtvm_end_time);
  uint64_t ns_spent = timing_cycles_to_ns(cycles);
  *elapsed_time_seconds = ns_spent / (double)1e9;
  g_microtvm_timer_running = 0;
  return kTvmErrorNoError;
}

// Ring buffer used to store data read from the UART on rx interrupt.
// This ring buffer size is only required for testing with QEMU and not for physical hardware.
#define RING_BUF_SIZE_BYTES (TVM_CRT_MAX_PACKET_SIZE_BYTES + 100)
RING_BUF_ITEM_DECLARE_SIZE(uart_rx_rbuf, RING_BUF_SIZE_BYTES);

// UART interrupt callback.
void uart_irq_cb(const struct device* dev, void* user_data) {
  uart_irq_update(dev);
  if (uart_irq_is_pending(dev)) {
    struct ring_buf* rbuf = (struct ring_buf*)user_data;
    if (uart_irq_rx_ready(dev) != 0) {
      uint8_t* data;
      uint32_t size;
      size = ring_buf_put_claim(rbuf, &data, RING_BUF_SIZE_BYTES);
      int rx_size = uart_fifo_read(dev, data, size);
      // Write it into the ring buffer.
      g_num_bytes_in_rx_buffer += rx_size;

      if (g_num_bytes_in_rx_buffer > RING_BUF_SIZE_BYTES) {
        TVMPlatformAbort((tvm_crt_error_t)0xbeef3);
      }

      if (rx_size < 0) {
        TVMPlatformAbort((tvm_crt_error_t)0xbeef1);
      }

      int err = ring_buf_put_finish(rbuf, rx_size);
      if (err != 0) {
        TVMPlatformAbort((tvm_crt_error_t)0xbeef2);
      }
      // CHECK_EQ(bytes_read, bytes_written, "bytes_read: %d; bytes_written: %d", bytes_read,
      // bytes_written);
    }
  }
}

#ifdef FVP
#define UART0_BASE 0x59303000
#define UART0_STATE (UART0_BASE + 0x04)
#define UART0_CTRL (UART0_BASE + 0x08)
#define UART0_INTCLEAR (UART0_BASE + 0x0C)
#define UART0_BAUDDIV (UART0_BASE + 0x10)
#define OVERRUN_IRQ 48  /* device uses IRQ 48 */
#define OVERRUN_PRIO 2  /* device uses interrupt priority 2 */
#define OVERRUN_ARG 0   /* argument passed to isr()*/
#define OVERRUN_FLAGS 0 /* IRQ flags. Unused on non-x86 */

void overrun_isr(void* arg) {
  *(uint32_t*)(UART0_STATE) = (uint32_t)0x00;     // clear overrun
  *(uint32_t*)(UART0_INTCLEAR) = (uint32_t)0x08;  // clear overrun
}

void overrun_init(void) {
  *(uint32_t*)(UART0_CTRL) = (uint32_t)0x2b;  // enable overrun interrupt
  IRQ_CONNECT(OVERRUN_IRQ, OVERRUN_PRIO, overrun_isr, OVERRUN_ARG, OVERRUN_FLAGS);
  irq_enable(OVERRUN_IRQ);
}

// Used to initialize the UART receiver.
void uart_rx_init(struct ring_buf* rbuf, const struct device* dev) {
  *(uint32_t*)(UART0_BAUDDIV) = (uint32_t)0xFF;  // set baudrate
  uart_irq_callback_user_data_set(dev, uart_irq_cb, (void*)rbuf);
  uart_irq_rx_enable(dev);
  overrun_init();
}
#else
// Used to initialize the UART receiver.
void uart_rx_init(struct ring_buf* rbuf, const struct device* dev) {
  uart_irq_callback_user_data_set(dev, uart_irq_cb, (void*)rbuf);
  uart_irq_rx_enable(dev);
}
#endif

// The main function of this application.
extern void __stdout_hook_install(int (*hook)(int));
void main(void) {
#ifdef CONFIG_LED
  int ret;
  led0_pin = device_get_binding(LED0);
  if (led0_pin == NULL) {
    for (;;)
      ;
  }
  ret = gpio_pin_configure(led0_pin, LED0_PIN, GPIO_OUTPUT_ACTIVE | LED0_FLAGS);
  if (ret < 0) {
    TVMPlatformAbort((tvm_crt_error_t)0xbeef4);
  }
  gpio_pin_set(led0_pin, LED0_PIN, 1);
#endif

  // Claim console device.
  tvm_uart = device_get_binding(DT_LABEL(DT_CHOSEN(zephyr_console)));
  uart_rx_init(&uart_rx_rbuf, tvm_uart);

  // Initialize system timing. We could stop and start it every time, but we'll
  // be using it enough we should just keep it enabled.
  timing_init();
  timing_start();

#ifdef FVP
  // initialize semihosting
  uart_log("init for a long time...\n");
  init_semihosting();
  uart_log("microTVM Zephyr runtime - running\n");
#endif

  // Initialize microTVM RPC server, which will receive commands from the UART and execute them.
  microtvm_rpc_server_t server = MicroTVMRpcServerInit(write_serial, NULL);
  TVMLogf("microTVM Zephyr runtime - running");
#ifdef CONFIG_LED
  gpio_pin_set(led0_pin, LED0_PIN, 0);
#endif

  // The main application loop. We continuously read commands from the UART
  // and dispatch them to MicroTVMRpcServerLoop().
  while (true) {
#ifdef FVP
    uint8_t data[128];
    uint32_t bytes_read = read_serial(data, 128);
#else
    uint8_t* data;
    unsigned int key = irq_lock();
    uint32_t bytes_read = ring_buf_get_claim(&uart_rx_rbuf, &data, RING_BUF_SIZE_BYTES);
#endif
    if (bytes_read > 0) {
      uint8_t* ptr = data;
      size_t bytes_remaining = bytes_read;
      while (bytes_remaining > 0) {
        // Pass the received bytes to the RPC server.
        tvm_crt_error_t err = MicroTVMRpcServerLoop(server, &ptr, &bytes_remaining);
        if (err != kTvmErrorNoError && err != kTvmErrorFramingShortPacket) {
          TVMPlatformAbort(err);
        }
#ifdef FVP
      }
    }
#else
        g_num_bytes_in_rx_buffer -= bytes_read;
        if (g_num_bytes_written != 0 || g_num_bytes_requested != 0) {
          if (g_num_bytes_written != g_num_bytes_requested) {
            TVMPlatformAbort((tvm_crt_error_t)0xbeef5);
          }
          g_num_bytes_written = 0;
          g_num_bytes_requested = 0;
        }
      }
      int err = ring_buf_get_finish(&uart_rx_rbuf, bytes_read);
      if (err != 0) {
        TVMPlatformAbort((tvm_crt_error_t)0xbeef6);
      }
    }
    irq_unlock(key);
#endif
  }

#ifdef CONFIG_ARCH_POSIX
  posix_exit(0);
#endif
}
