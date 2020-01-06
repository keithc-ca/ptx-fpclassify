JAVA_HOME ?= ./jdk

all : bin/ieee754/FPClassify.class

clean :
	rm -rf bin

test :
	$(JAVA_HOME)/bin/java -cp bin ieee754.FPClassify

bin/ieee754/FPClassify.class : src/ieee754/FPClassify.java
	@mkdir -p $(@D)
	$(JAVA_HOME)/bin/javac -d bin $<
