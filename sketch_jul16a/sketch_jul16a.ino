#include <Wire.h>
#include <Adafruit_MMC56x3.h>

#define TCAADDR 0x70  // Default I2C address for TCA9548A

void tcaselect(uint8_t i) {
  if (i > 7) return;
  Wire.beginTransmission(TCAADDR);
  Wire.write(1 << i);
  Wire.endTransmission();
}


 // Create an array for 8 sensor objects, each with a unique ID
Adafruit_MMC5603 mmc[8] = {
  Adafruit_MMC5603(0),
  Adafruit_MMC5603(1),
  Adafruit_MMC5603(2),
  Adafruit_MMC5603(3),
  Adafruit_MMC5603(4),
  Adafruit_MMC5603(5),
  Adafruit_MMC5603(6),
  Adafruit_MMC5603(7)
};

void setup() {
  Serial.begin(115200);
  Wire.begin();
  
  Serial.println("Initializing all MMC5603 sensors:");
  for (int i = 0; i < 8; i++) {
    // if(  i==1)
    // continue;
    tcaselect(i);
    delay(5);
    if (!mmc[i].begin(MMC56X3_DEFAULT_ADDRESS, &Wire)) {
      Serial.print("Sensor ");
      Serial.print(i);
      Serial.println(" not detected.");
    } else {
      Serial.print("Sensor ");
      Serial.print(i);
      Serial.println(" initialized.");
    }
    delay(5);
  }
  Serial.println("All sensors initialization attempted.");
}

void loop() {
  for (int i = 0; i < 8; i++ ) {
    // if(i==1)
    // continue;
    tcaselect(i);
    // delay(10);
    sensors_event_t event;
    if (mmc[i].getEvent(&event)) {
      Serial.print(i); Serial.print(",");
      Serial.print(event.magnetic.x); Serial.print(",");
      Serial.print(event.magnetic.y); Serial.print(",");
      Serial.println(event.magnetic.z);
    } else {
      Serial.print(i); Serial.println(",fail");
    }
    delay(5);
  }
  Serial.println("BREAK");
  delay(5);
}