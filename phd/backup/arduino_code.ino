#include <Wire.h>
#include <U8g2lib.h>
#include <vector>

#define BUTTON_1 1
#define BUTTON_2 2
// #define LED_2 3
#define LED_3 4
#define I2C_SDA_PIN 5   // 3 // 5 // 17
#define I2C_SCL_PIN 6   // 2 // 6 // 18
#define MCU_SDA 7
#define MCU_SCL 8
#define DEVICE_ADDRESS 0x5D  // 另一个设备的I2C地址 5D / 14
#define RAW_ADDRESS 0X8B98
#define CAL_ADDRESS 0X81C0

uint16_t channelDrive;
uint16_t channelSensor;
std::vector<uint16_t> data;  // Declare a pointer
bool flag_read = true;
byte data1;

U8G2_SSD1316_128X32_F_SW_I2C u8g2(U8G2_R2, MCU_SCL, MCU_SDA);
// GREEN ----> U8G2_SSD1316_128X32_F_SW_I2C u8g2(U8G2_R2, I2C_SCL_PIN, I2C_SDA_PIN);

void setup() {
  generalInit();
  monitorInit();
  GT9110Init();
  channelInit();
}

void generalInit() {
  pinMode(BUTTON_1, INPUT);
  pinMode(BUTTON_2, INPUT);
  // pinMode(LED_2, OUTPUT);
  pinMode(LED_3, OUTPUT);
  Serial.begin(9600);
  Serial.setTimeout(0);  // Set a shorter timeout for serial reading
  delay(100);
}

void monitorInit() {
  u8g2.begin();
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_ncenB12_tr);
  u8g2.drawStr(15, 20, "Hi Ping");
  u8g2.sendBuffer();
  delay(100);
}

void GT9110Init() {
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);  // 初始化I2C总线，指定SDA和SCL引脚
  Wire.setClock(400000);                 // 设置I2C时钟速率为400 kHz
  delay(100);
}

void channelInit() {
  Wire.beginTransmission(DEVICE_ADDRESS);
  Wire.write(0x80);  // 发送高位字节
  Wire.write(0x62);  // 发送低位字节
  Wire.endTransmission();

  Wire.requestFrom(DEVICE_ADDRESS, 3);
  if (Wire.available()) {
    data1 = Wire.read();  // 读取数据
    channelDrive = data1 % 32;

    data1 = Wire.read();  // 读取数据
    channelDrive = channelDrive + data1 % 32;

    data1 = Wire.read();  // 读取数据
    channelSensor = int(data1 / 16) + data1 % 16;

    Serial.println(channelDrive);
    Serial.println(channelSensor);

    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_ncenB08_tr);
    u8g2.drawStr(0, 10, "GT9110 is connected.");
    u8g2.drawStr(0, 21, "Drive: Sensor:");
    u8g2.drawStr(10, 32, String(channelDrive).c_str());
    u8g2.drawStr(50, 32, String(channelSensor).c_str());
    u8g2.sendBuffer();

    for (int i = 0; i < channelDrive * channelSensor + 4; i++) {
      data.push_back(999);
    }

    delay(1000);

    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_ncenB08_tr);
    u8g2.drawStr(0, 10, "Size is obtained.");
    u8g2.drawStr(0, 21, "Drive: Sensor:");
    u8g2.drawStr(10, 32, String(data.size()).c_str());
    u8g2.sendBuffer();
  }
}

void updateCal() {
  Wire.beginTransmission(DEVICE_ADDRESS);
  Wire.write(0x80);  // 发送高位字节
  Wire.write(0x40);  // 发送低位字节
  Wire.write(0x03);  // 发送更新指令
  Wire.endTransmission();
  delay(100);
}

void channelCheck() {
  Wire.beginTransmission(DEVICE_ADDRESS);
  Wire.write(0x80);  // 发送高位字节
  Wire.write(0x62);  // 发送低位字节
  Wire.endTransmission();

  Wire.requestFrom(DEVICE_ADDRESS, 3);
  if (Wire.available()) {
    data1 = Wire.read();  // 读取数据
    channelDrive = data1 % 32;

    data1 = Wire.read();  // 读取数据
    channelDrive = channelDrive + data1 % 32;

    data1 = Wire.read();  // 读取数据
    channelSensor = int(data1 / 16) + data1 % 16;

    uint16_t channelData[6];
    channelData[0] = 55555;
    channelData[1] = 55555;
    channelData[2] = channelDrive;
    channelData[3] = channelSensor;
    channelData[4] = 44444;
    channelData[5] = 44444;

    Serial.print("channelDrive: ");
    Serial.print(channelDrive);
    Serial.print("  ");
    Serial.print("channelSensor: ");
    Serial.println(channelSensor);
  }
}

void sendData(uint16_t address) {
  int number = 2;

  Wire.beginTransmission(DEVICE_ADDRESS);
  Wire.write((address >> 8) & 0xFF);  // 发送高位字节
  Wire.write((address) & 0xFF);       // 发送低位字节
  Wire.endTransmission();

  Wire.requestFrom(DEVICE_ADDRESS, 2 * channelDrive * channelSensor);

  data[0] = 55555;
  data[1] = 55555;
  data[2 + channelDrive * channelSensor] = 44444;
  data[3 + channelDrive * channelSensor] = 44444;

  for (int i = 0; i < channelDrive; i++) {
    for (int j = 0; j < channelSensor; j++) {
      data1 = Wire.read();  // 读取数据
      uint16_t value = data1 * 256;
      data1 = Wire.read();  // 读取数据
      value = value + data1;
      data[number] = value;
      number++;
    }
  }

  // Print the data array contents to the serial monitor
  for (int i = 0; i < number + 2; i++) {
    Serial.print(data[i]);
    Serial.print(" ");
  }
  Serial.println();
}

void sendData_Robust(uint16_t baseAddress) {
  // Calculate the total number of 16-bit values to read.
  uint16_t totalValues = channelDrive * channelSensor;
  uint16_t totalBytes = totalValues * 2;

  // --- CRITICAL DIAGNOSTIC ---
  // Let's see how much data we're dealing with. This is very important.
  // Serial.println("--------------------");
  // Serial.print("Total Drive Channels: ");
  // Serial.println(channelDrive);
  // Serial.print("Total Sensor Channels: ");
  // Serial.println(channelSensor);
  // Serial.print("Calculated values to read: ");
  // Serial.println(totalValues);
  // Serial.println("--------------------");
  // delay(100); // Allow time to print before starting the heavy lifting.

  // Define a manageable chunk size for I2C reads (in bytes).
  // 64 bytes is a safe value that is well within the buffer limits.
  const int I2C_CHUNK_SIZE = 64;
  uint16_t bytesRead = 0;

  // CHANGED: Use print() instead of println() to stay on the same line.
  Serial.print("55555 55555 ");

  while (bytesRead < totalBytes) {
    // Calculate the current memory address to read from the sensor
    uint16_t currentAddress = baseAddress + bytesRead;

    // 1. Set the register pointer on the I2C device
    Wire.beginTransmission(DEVICE_ADDRESS);
    Wire.write((currentAddress >> 8) & 0xFF);  // Send high byte of address
    Wire.write(currentAddress & 0xFF);         // Send low byte of address
    byte error = Wire.endTransmission();

    if (error != 0) {
      Serial.print("Error setting I2C address pointer. Code: ");
      Serial.println(error);
      return;  // Stop if we can't communicate
    }

    // 2. Request one chunk of data
    uint16_t bytesToRequest = I2C_CHUNK_SIZE;

    // If the remaining data is less than a full chunk, request only what's left.
    if (totalBytes - bytesRead < I2C_CHUNK_SIZE) {
      bytesToRequest = totalBytes - bytesRead;
    }

    Wire.requestFrom(DEVICE_ADDRESS, bytesToRequest);

    // 3. Process the received chunk
    while (Wire.available() >= 2) {
      uint8_t highByte = Wire.read();
      uint8_t lowByte = Wire.read();
      uint16_t value = (highByte << 8) | lowByte;

      Serial.print(value);  // This correctly prints the data on the same line
      Serial.print(" ");

      bytesRead += 2;
    }

    // This small delay is crucial. It gives the serial buffer time to send
    // and allows other system tasks (like WiFi, etc.) to run.
    // delay(5);
  }

  // CHANGED: Use print() and remove the extra newline character '\n'.
  Serial.print("44444 44444");

  // ADDED: A final println() to move to the next line after the output is complete.
  Serial.println();

  Serial.flush();  // Ensure all data is sent
  // Serial.println("--- READ COMPLETE ---");
}

void loop() {
  // unsigned long ctime = millis();

  if (Serial.available() > 0) {
    String response = "Current time: " + String(millis());
    String request = Serial.readString();  // Read the input from serial monitor

    // Process the request here
    if (request.indexOf("readRaw") != -1) {
      // sendData(RAW_ADDRESS);
      sendData_Robust(RAW_ADDRESS);  // <-- CALL THE NEW FUNCTION
      flag_read = true;
    }

    if (request.indexOf("stop") != -1) {
      flag_read = false;
    }

    if (request.indexOf("readCal") != -1) {
      flag_read = false;
      // sendData(CAL_ADDRESS);
      sendData_Robust(CAL_ADDRESS);  // <-- CALL THE NEW FUNCTION
    }

    if (request.indexOf("updateCal") != -1) {
      updateCal();
      // sendData(CAL_ADDRESS);
      sendData_Robust(CAL_ADDRESS);  // <-- CALL THE NEW FUNCTION
    }

    if (request.indexOf("channelCheck") != -1) {
      channelCheck();
    }
  }

  // Serial.println("Disconnected");
}
