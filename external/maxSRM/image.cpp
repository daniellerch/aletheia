/******************************************************************************************
  Class for reading and writing PGM grayscale 8-bit images.                       *
  Written by: Tomas Filler (tomas.filler@gmail.com, http://dde.binghamton.edu/filler/)    *
                                                                                          *
  Feel free to modify it according to your needs.                                         *
******************************************************************************************/

#include "image.h"
#include "exception.cpp"
#include <fstream>
#include <string>
#include <cstring>

/*
 * Simple constructor
 */
image::image() {
    this->pixels = 0;
}

/* Constructor from existing data
 */
image::image(int width, int height, unsigned char *pixels) {
    this->width = width;
    this->height = height;
    this->pixels = new unsigned char[width*height];
    memcpy(this->pixels, pixels, width*height*sizeof(unsigned char));
}

/* Read PGM image from file and fills width and height appropriately
 *   @param fileName path to the image to load
 */
void image::load_from_pgm(std::string file_name) {
    
    std::ifstream file;
    char c1, c2;
    int max_level;
    char *buffer;

    file.open(file_name.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) { throw exception("[read_pgm_file] File "+file_name+" could not be opened for reading."); }
    file >> c1 >> c2;
    if ( (c1!='P') || (c2!='5') ) { throw exception("[read_pgm_file] File "+file_name+" starts with different characters than 'P5'."); }
    file >> width >> height >> max_level;
    buffer = new char[width*height];
    if (pixels!=0) delete[] pixels;
    pixels = new unsigned char[width*height];
    file.read(buffer,1); //junk
    file.read(buffer, width*height); // read binary data. PGM stores image by rows and I use columnwise ordering
    file.close();
    for(int i=0;i<height;i++)
        for(int j=0;j<width;j++) {
            unsigned char c = buffer[i*width+j];
            pixels[j*height+i] = c;
        }
    delete[] buffer;
}

void image::write_to_pgm(std::string file_name) {

    std::ofstream file;

    file.open (file_name.c_str(), std::ios::out | std::ios::binary);
    if (!file.is_open()) { throw exception("[write_pgm_file] File "+file_name+" could not be opened for writing"); }
    file << "P5\n" << width << " " << height << "\n255\n";
    for(int i=0;i<height;i++)
        for(int j=0;j<width;j++)
            file.write((char*)(pixels+j*height+i),1);
    file.close();
}

image::~image() {
    if (this->pixels!=0) { delete[] this->pixels; }
}
