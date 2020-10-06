/*
 * Team Id: eYRC#64
 * Author List: Vishnukhanth, ShivaKumar
 * Filename: main.c
 * Theme: Supply Bot
 * Functions: void adc_pin_config(void),
 void init_adc(void), void init_motors(void), void init_timers(void),
 void init_buzzer(void), void init_servo(void),
 uint16_t adc_conversion(unsigned char channel), void set_speed(unsigned char left, unsigned char right), void forward(void), void backward(void), void stop(void),
 void calculate_PID(void),
 void left_forward_right_backward(void),
 void adjust_position(),
 void buzzer_beep_twice(), void buzzer_beep_1sec(), void init_uart0(), void acknowledge(),
 void uart_rx(), void xbee_comm(), void hitting_mechanism(), void init_devices(void),
int main(void)
 
 * Global Variables: 
 * variables>
 */

#define F_CPU 16000000UL

#define PIN_LEFT_SENSOR PC0
#define PIN_MIDDLE_SENSOR PC1
#define PIN_RIGHT_SENSOR PC2
#define PIN_ENA PD6
#define PIN_ENB PB3
#define PIN_IN1 PD7
#define PIN_IN2 PB0
#define PIN_IN3 PB4
#define PIN_IN4 PD5
#define PIN_buzzer PD3
#define PIN_servo PB1


#include <avr/io.h>
#include <util/delay.h>

uint16_t adc_value_left;
uint16_t adc_value_middle;
uint16_t adc_value_right;
uint16_t adc_value;

unsigned thresh = 200;
int flag = 0, i = 0, count = 0, city_flag = 0, direction_pointer = 4, city_count;
char path[8]; //= {'2', '3', '2', '3', 'c', 'c' ,'c', 'c'};


signed int total_error = 0;
signed int last_error = 0;
signed int error;
//float kp = 10;
float kp = 15;
float ki = 0.0005;
float kd = 3;
//float ki = 0;
//float kd = 0;
float pid;


/* Setting PC0, PC1 and PC2 as inputs for left, middle and right sensor respectively
   Pull-up resistor not enabled since ADC won't give correct output
*/

/*
 * Function Name: adc_pin_config
 * Input: none
 * Output: none
 * Logic: Initializes the pins of port C for ADC use,Setting PC0, PC1 and PC2 as inputs for left, middle and right sensor respectively
 * Pull-up resistor not enabled since ADC won't give correct output
 * Example Call: adc_pin_config()
*/

void adc_pin_config(void)
{
	DDRC &= ~(1 << PIN_LEFT_SENSOR | 1 << PIN_MIDDLE_SENSOR | 1 << PIN_RIGHT_SENSOR);
	PORTC &= ~(1 << PIN_LEFT_SENSOR | 1 << PIN_MIDDLE_SENSOR | 1 << PIN_RIGHT_SENSOR);	
}

/* 1) Disabling analog comparator.
   2) Enabling external reference voltage AVcc (5V).
   3) Enabling ADC, setting Pre-scaler of 64 and setting ADC interrupt enable
*/


/*
 * Function Name: adc_init
 * Input: none
 * Output: none
 * Logic: Initializes registers values for ADC use,Disabling analog comparator,Enabling external reference voltage AVcc(5V),Enabling ADC, setting Pre-scaler of 64 and setting ADC interrupt enable
 * Example Call: adc_init();
*/
void init_adc(void)
{
	ACSR |= (1 << ACD); 
	ADMUX |= (1 << REFS0 );
	ADCSRA |= ((1 << ADEN) |  (1 << ADPS2 | 1 << ADPS1) | 1 << ADIE);	
}

// initialization of motor pins.
/*
 * Function Name: init_motors
 * Input: none
 * Output: none
 * Logic: Initializes the pins of port B, PB0 and PB4 and port D,PD7 and PD5 for motor use, and uses pins PD6 and PB3 for PWM generation using 8-bit timer0 and timer1 in fast PWM MODE.
 * Example Call: init_motors()
*/
void init_motors(void)
{
	DDRB |= (1 << PIN_ENB | 1 << PIN_IN2 | 1 << PIN_IN3);
	PORTB = 0x00;
	DDRD |= (1 << PIN_ENA | 1 << PIN_IN1 | 1 << PIN_IN4);
	PORTD = 0x00;
}
/*
 * Function Name: init_timers
 * Input: none
 * Output: none
 * Logic: Initializes 8-bit timer0 and 8-bit timer2 for velocity control of motors and 16-bit timer1 for servo control
 * Example Call: init_timers()
*/
// initialization of timers...
void init_timers(void)
{
	// timer 0 initialization
	TCCR0A |= (1 << COM0A1 | 1 << WGM01 | 1 << WGM00);
	TCCR0B |= (1 << CS00 | 1 << CS01);
	TIMSK0 |= (1 << OCIE0A | 1 << TOIE0);
	// timer 1 initialization
	TCCR1A |= 1<<WGM11 | 1<<COM1A1 | 1<<COM1A0;
	TCCR1B|=(1<<WGM13)|(1<<WGM12)|(1<<CS11)|(1<<CS10);
	ICR1 = 4999;
	// timer 2 initialization
	TCCR2A |= (1 << COM2A1 | 1 << WGM21 | 1 << WGM20);
	TCCR2B |= (1 << CS22);
	TIMSK2 |= (1 << OCIE2A | 1 << TOIE2);	
}


/*
 * Function Name: init_buzzer
 * Input: none
 * Output: none
 * Logic: Initializes the pin of port D, PD3 for buzzer
 * Example Call: init_buzzer()
*/

void init_buzzer(void)
{
	DDRD |= (1 << PIN_buzzer);
	PORTD |= (1 << PIN_buzzer);
}


/*
 * Function Name: init_servo
 * Input: none
 * Output: none
 * Logic: Initializes the pin of port B, PB1 for buzzer
 * Example Call: init_servo()
*/
void init_servo(void)
{
	DDRB |= (1 << PIN_servo);
}
/* Function for ADC conversion.
   Argument : channel
*/


/*
 * Function Name: adc_conversion
 * Input: channel
 * Output: adc_value
 * Logic: converts the analog value into digital value ranging from 0-1023 using 10-bit analog to digital converter .
 * Example Call: adc_conversion()
*/


uint16_t adc_conversion(unsigned char channel)
{
	// getting last three bits of channel
	channel = channel & 0b00000111;
	// setting channel ADMUX.
	ADMUX = 0x40 | channel;
	// start ADC 
	ADCSRA |= (1 << ADSC);
	// waiting until the conversion completes.
	while((ADCSRA & (1 << ADIF)) == 0);
	// getting the conversion result.
	uint8_t Low = ADCL;
	// extracting 10 bit adc value
	adc_value = ADCH <<8 | Low;
	ADCSRA |= (1 << ADIF);
	
	return adc_value;		
}

// setting the speed of left and right motors
/*
 * Function Name: set_speed
 * Input: left, right
 * Output: none
 * Logic: Assignes speed for left motor and right motor by generating PWM wave at OCR2A and OCR0A
 * Example set_speed()
*/
void set_speed(unsigned char left, unsigned char right)
{
	OCR2A = (unsigned char)left;
	OCR0A = (unsigned char)right;
}

// forward motion
/*
 * Function Name: foreward
 * Input: none
 * Output: none
 * Logic: will set logic 1 to pins PD6,PD7,PB3 and PB4 for rotating motors in the foreward direction.
 * Example Call: foreward()
*/
void forward(void)
{
	PORTD = 0xC8;
	PORTB = 0x18;
}

// backward motion
/*
 * Function Name: backward
 * Input: none
 * Output: none
 * Logic: will set logic 1 to pins PD5,PD6,PB0 and PB3 for rotating motors in the backward direction.
 * Example Call: backward()
*/
void backward(void)
{
	PORTD = 0x68;
	PORTB = 0x09;
}

/*
 * Function Name: stop
 * Input: none
 * Output: none
 * Logic: will set logic 0 to pins of port D to 0 except PD3 and pins of port B to stop the motors
 * Example Call: stop()
*/

void stop(void)
{
	PORTD = 0x08;
	PORTB = 0x00;
}

/*
 * Function Name: calculate_PID
 * Input: none
 * Output: none
 * Logic: Assignes random values to kp,ki and kd initially and fine tunes it's values until the robot smoothly follows whiteline .
 * Example call:calculate_PID()
*/

void calculate_PID(void)
{
	total_error += error;
	pid = error*kp + total_error*ki + (error-last_error)*kd;
	pid = (int) pid;
	last_error = error;
}

/*
 * Function Name: left_forward_right_backward
 * Input: none
 * Output: none
 * Logic: Assignes random values to kp,ki and kd initially and fine tunes it's values until the robot smoothly follows whiteline .
 * Example call: left_forward_right_backward()
*/
void left_forward_right_backward(void)
{
	PORTD = 0x48;
	PORTB = 0x19;
}

void adjust_position()
{
	set_speed(70, 70);
	left_forward_right_backward();
	_delay_ms(200);
	stop();
}

void buzzer_beep_twice()
{
	for(int j=0; j<2; j++)
	{
		PORTD &= ~(1 << PIN_buzzer);
		_delay_ms(500);
		PORTD |= (1 << PIN_buzzer);
		_delay_ms(500);
	}
}

void buzzer_beep_1sec()
{
	PORTD &= ~(1 << PIN_buzzer);
	_delay_ms(1000);
	PORTD |= (1 << PIN_buzzer);
}

void buzzer_beep_5sec()
{
	PORTD &= ~(1 << PIN_buzzer);
	_delay_ms(5000);
	PORTD |= (1 << PIN_buzzer);
}

void init_uart0()
{
	UCSR0B |= (1 << RXEN0 | 1 << TXEN0 | 1 << RXCIE0);
	UCSR0C |= (1 << UCSZ00 | 1 << UCSZ01 );
	UBRR0L = 103;
}

void acknowledge()
{
	if (path[i-1] != '0')
	{
		while ((UCSR0A & (1 << UDRE0))==0);
		UDR0 = 'y';
	}
}

void uart_rx()
{
	while((UCSR0A & (1 << RXC0))==0);
	path[i] = UDR0;
	acknowledge();
	i++;
}

void xbee_comm()
{
	for(int j = 0; j <= 7; j++)
	uart_rx();
}


void hitting_mechanism()
{
	OCR1A = 4690;
	_delay_ms(1000);
	OCR1A = 4640;
}

// initialization of devices
void init_devices(void)
{
	adc_pin_config();
	init_adc();
	init_motors();
	init_timers();
	init_buzzer();
	init_uart0();
	init_servo();
}

int main(void)
{
	init_devices();
	
	xbee_comm();
	
	i = 0;
	city_count = path[i]-'0';

	if (path[direction_pointer] == 'c')
		flag = 1;
	else
		flag = 0;
		
	set_speed(80, 80);
	if (flag == 1)
		forward();
	else
		backward();
	
	_delay_ms(300);

    while (1) 
    {
		adc_value_left = adc_conversion(0);
		adc_value_middle = adc_conversion(1);
		adc_value_right = adc_conversion(2);
		
		if (((adc_value_left > thresh) && (adc_value_middle < thresh) && (adc_value_right > thresh)))
		{
			city_flag = 0;
			error = 0;
			calculate_PID();
			set_speed(70+pid, 70-pid);
			//set_speed(80, 80);
			if (flag == 1)	
				forward();
			else
				backward();
		}
		if (((adc_value_left < thresh) && (adc_value_middle > thresh) && (adc_value_right > thresh)))
		{
			city_flag = 0;
			error = -1;
			calculate_PID();
			set_speed(70+pid,70-pid);
			// set_speed(60, 80);
			if (flag == 1)
				forward();
			else
				backward();
		}
		if (((adc_value_left >thresh) && (adc_value_middle > thresh) && (adc_value_right < thresh)))
		{
			city_flag = 0;
			error = 1;
			calculate_PID();
			set_speed(70+pid, 70-pid);
			// set_speed(80,60);
			if (flag == 1)
				forward();
			else
				backward();
		}

		if (((adc_value_left < thresh) && (adc_value_middle < thresh)) || ((adc_value_middle < thresh) && (adc_value_right < thresh)))
		{
			
			if (count != city_count && city_flag == 0)
			{
				count++;
				city_flag = 1;
			}
			
			if (count == city_count)
			{
				stop();
				if (i != 3)
				{
					adjust_position();
					buzzer_beep_twice();
					hitting_mechanism();
					_delay_ms(5000);
					buzzer_beep_1sec();
				}
				direction_pointer++;
				count = 0;
				i++;
				city_count = path[i]-'0';
				_delay_ms(2000);
				error = 0;
				calculate_PID();
				set_speed(70+pid,70-pid);
				// set_speed(80, 80);

				if (i < 4)
				{
					if (path[direction_pointer] == 'c')
					{
						flag = 1;
						forward();
					}			
					else
					{
						flag = 0;
						backward();
					}
					_delay_ms(250);
				}
				
				else
				{
					stop();
					buzzer_beep_5sec();
					while(1);
				}
			}
		}
    }
}









