#include <Wire.h>
#include <Adafruit_MMC56x3.h>

#define TCAADDR 0x70  // I2C address of TCA9548A multiplexer
int sensor = 1;
// Multiplexer channel selector
void tcaselect(uint8_t i) {
  if (i > sensor) return;
  Wire.beginTransmission(TCAADDR);
  Wire.write(1 << i); // Set only the i-th bit high
  Wire.endTransmission();
}

// Create MMC5603 sensor object
Adafruit_MMC5603 mmc = Adafruit_MMC5603(12345);

void setup() {
  Serial.begin(115200);
  Wire.begin();

  tcaselect(sensor);
  // delay(10);

  if (!mmc.begin(MMC56X3_DEFAULT_ADDRESS, &Wire)) {
    Serial.println("Sensornot detected!");
    while (1);
  }

  Serial.println("Sensor  initialized!");
}

void loop() {
  tcaselect(sensor);
  // delay(5);

  sensors_event_t event;
  if (mmc.getEvent(&event)) {
    Serial.print(event.magnetic.x); Serial.print(",");
    Serial.print(event.magnetic.y); Serial.print(",");
    Serial.println(event.magnetic.z);
  } else {
    Serial.println("Sensorread failed!");
  }

  delay(10);
}
