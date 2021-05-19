/******************************************************************************************
  Class for reading and writing PNG and PGM grayscale 8-bit images.                       *
  Written by: Tomas Filler (tomas.filler@gmail.com, http://dde.binghamton.edu/filler/)    *
                                                                                          *
  Feel free to modify it according to your needs.                                         *
******************************************************************************************/
#ifndef IMAGE_H_
#define IMAGE_H_

#include <string>

class image {
public:
    int width, height;
    unsigned char *pixels;
    
    image();
    image(int width, int height, unsigned char *pixels);
    void load_from_pgm(std::string file_name);
    void write_to_pgm(std::string file_name);
    ~image();
};

#endif
