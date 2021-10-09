/* 
 * Button Example for Rosserial
 */

#include <ros.h>
#include <std_msgs/Bool.h>


ros::NodeHandle nh;

//std_msgs::Bool pushed_msg;
//ros::Publisher pub_button("pushed", &pushed_msg);

std_msgs::Bool pushed_msg_red;
ros::Publisher pub_button_red("RED_Emergency_Stop", &pushed_msg_red);
std_msgs::Bool pushed_msg_yellow;
ros::Publisher pub_button_yellow("YELLOW_Manual_Control", &pushed_msg_yellow);
std_msgs::Bool pushed_msg_green;
ros::Publisher pub_button_green("GREEN_Start_DDPG", &pushed_msg_green);



//const int button_pin = 12;
//const int led_pin = 13;

const int button_pin_red = 8;
const int button_pin_yellow = 10; 
const int button_pin_green = 12;
const int led_pin = 13;

//
//bool last_reading;
//bool published = true;

bool last_reading_red;
bool last_reading_yellow;
bool last_reading_green;
bool published_red = true;
bool published_yellow = true;
bool published_green = true;

long last_debounce_time=0;
long debounce_delay=50;

void setup()
{
  nh.initNode();
//  nh.advertise(pub_button);
  nh.advertise(pub_button_red);
  nh.advertise(pub_button_yellow);
  nh.advertise(pub_button_green);
  
  //initialize an LED output pin 
  //and a input pin for our push button
  pinMode(led_pin, OUTPUT);
//  pinMode(button_pin, INPUT);
  pinMode(button_pin_red, INPUT);
  pinMode(button_pin_yellow, INPUT);
  pinMode(button_pin_green, INPUT);
  
  //Enable the pullup resistor on the button
//  digitalWrite(button_pin, HIGH);
  digitalWrite(button_pin_red, HIGH);
  digitalWrite(button_pin_yellow, HIGH);
  digitalWrite(button_pin_green, HIGH);
  
  //The button is a normally button
//  last_reading = ! digitalRead(button_pin);
  last_reading_red = ! digitalRead(button_pin_red);
  last_reading_yellow = ! digitalRead(button_pin_yellow);
  last_reading_green = ! digitalRead(button_pin_green); 
}

void loop()
{
  
//  bool reading = ! digitalRead(button_pin);
  bool reading_red = ! digitalRead(button_pin_red);
  bool reading_yellow = ! digitalRead(button_pin_yellow);
  bool reading_green = ! digitalRead(button_pin_green);

//  if (last_reading!= reading){
//      last_debounce_time = millis();
//      published = false;
//  }
  if (last_reading_red!= reading_red){
      last_debounce_time = millis();
      published_red = false;
  }
  if (last_reading_yellow!= reading_yellow){
      last_debounce_time = millis();
      published_yellow = false;
  }
  if (last_reading_green!= reading_green){
      last_debounce_time = millis();
      published_green = false;
  }
  
  //if the button value has not changed for the debounce delay, we know its stable
//  if ( !published && (millis() - last_debounce_time)  > debounce_delay) {
//    digitalWrite(led_pin, reading);
//    pushed_msg.data = reading;
//    pub_button.publish(&pushed_msg);
//    published = true;
//  }
  if ( !published_red && (millis() - last_debounce_time)  > debounce_delay) {
    digitalWrite(led_pin, reading_red);
    pushed_msg_red.data = reading_red;
    pub_button_red.publish(&pushed_msg_red);
    published_red = true;
  }
  if ( !published_yellow && (millis() - last_debounce_time)  > debounce_delay) {
    digitalWrite(led_pin, reading_yellow);
    pushed_msg_yellow.data = reading_yellow;
    pub_button_yellow.publish(&pushed_msg_yellow);
    published_yellow = true;
  }
  if ( !published_green && (millis() - last_debounce_time)  > debounce_delay) {
    digitalWrite(led_pin, reading_green);
    pushed_msg_green.data = reading_green;
    pub_button_green.publish(&pushed_msg_green);
    published_green = true;
  }
//
//  last_reading = reading;
  last_reading_red = reading_red;
  last_reading_yellow = reading_yellow;
  last_reading_green = reading_green;
  
  nh.spinOnce();
}
