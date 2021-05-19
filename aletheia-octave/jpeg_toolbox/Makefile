default:
	mkoctfile --mex jpeg_read.c -ljpeg
	mkoctfile --mex jpeg_write.c -ljpeg
	cp -f *.mex ../octave/
clean:
	rm -f *.o *.mex
