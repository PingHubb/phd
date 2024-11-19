#include <Wire.h>
#include <U8g2lib.h>
#include <vector>

#define BUTTON_1 1
#define BUTTON_2 2
#define LED_2 3
#define LED_3 4
#define I2C_SDA_PIN 5
#define I2C_SCL_PIN 6
#define MCU_SDA 7
#define MCU_SCL 8
#define DEVICE_ADDRESS 0x5D // 另一个设备的I2C地址 5D / 14
#define RAW_ADDRESS 0X8B98
#define CAL_ADDRESS 0X81C0
uint16_t channelDrive;
uint16_t channelSensor;
std::vector<uint16_t> data;  // Declare a pointer
bool flag_read = true;
byte data1;
int jj = 0;

U8G2_SSD1316_128X32_F_SW_I2C u8g2(U8G2_R2, MCU_SCL, MCU_SDA);  //  GREEN ---->   U8G2_SSD1316_128X32_F_SW_I2C u8g2(U8G2_R2, I2C_SCL_PIN, I2C_SDA_PIN);

void setup() {
  generalInit();
  monitorInit();
  GT9110Init();
  channelInit();
}

void generalInit() {
  pinMode(BUTTON_1, INPUT);
  pinMode(BUTTON_2, INPUT);
  pinMode(LED_2, OUTPUT);
  pinMode(LED_3, OUTPUT);
  Serial.begin(9600);
  Serial.setTimeout(0);  // Set a shorter timeout for serial reading

  delay(100);
}

void monitorInit() {
  u8g2.begin();
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_ncenB12_tr);
  u8g2.drawStr(15, 20, "Singa Viva");
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
  Wire.write((address)&0xFF);         // 发送低位字节
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

  // Print the count of printed elements
  // Serial.print("Number of elements printed: ");
  // Serial.println(jj);
  // jj = jj + 1;
}

void loop() {

  // unsigned long ctime = millis();

  Serial.print("hello");
  if (Serial.available() > 0) {
    String response = "Current time: " + String(millis());
    String request = Serial.readString();  // Read the input from serial monitor
    // Process the request here

    if (request.indexOf("readRaw") != -1) {
      sendData(RAW_ADDRESS);
      // unsigned long dtime = millis();
      // Serial.println("Time passed: ");
      // Serial.println(dtime - ctime);
      flag_read = true;
    }
    if (request.indexOf("stop") != -1) {
      flag_read = false;
    }
    if (request.indexOf("readCal") != -1) {


      flag_read = false;
      // for (int i = 0; i < 5; i++) {
        sendData(CAL_ADDRESS);
      // }
    }
    if (request.indexOf("updateCal") != -1) {
      updateCal();
      // for (int i = 0; i < 5; i++) {
        sendData(CAL_ADDRESS);
      // }
    }
    if (request.indexOf("channelCheck") != -1) {
      channelCheck();
    }
  }

  // Serial.println("Disconnected");
}