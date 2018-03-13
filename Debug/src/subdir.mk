################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/dataset.cpp \
../src/main.cpp \
../src/model.cpp \
../src/polya_fit_simple.cpp \
../src/tokenizer.cpp

OBJS += \
./src/dataset.o \
./src/main.o \
./src/model.o \
./src/polya_fit_simple.o \
./src/tokenizer.o

CPP_DEPS += \
./src/dataset.d \
./src/main.d \
./src/model.d \
./src/polya_fit_simple.d \
./src/tokenizer.d


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


