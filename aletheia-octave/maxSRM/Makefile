default:
	mkoctfile --mex SRM_matlab.cpp SRMclass.cpp submodel.cpp s.cpp image.cpp exception.cpp config.cpp
	cp -f SRM_matlab.mex ../octave/MAXSRM.mex
clean:
	rm -f *.o *.mex
