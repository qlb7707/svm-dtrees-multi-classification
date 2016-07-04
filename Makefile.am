AUTOMAKE_OPTIONS=foreign
bin_PROGRAMS=train_test
INCLUDES=`pkg-config --cflags opencv`
train_test_SOURCES=typedef.h basictypes.h checks.h checks.cpp rtc_common.h rtc_common.cpp flags.h flags.cpp flag_def.h common.h common.cpp train_test.cpp
LIBS=`pkg-config --libs opencv`
