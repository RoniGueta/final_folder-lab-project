CC=gcc
CFLAGS = -march=native -mtune=native -Ofast

extract_mvs: extract_mvs.c
	gcc -L/opt/ffmpeg/lib -I/opt/ffmpeg/include/ extract_mvs.c -lavcodec -lavformat -lavutil $(CFLAGS) -o extract_mvs

clean:
	rm extract_mvs