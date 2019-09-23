
### Statistical attacks to LSB replacement

LSB replacement staganographic methods, that is, methods that hide information replacing the least significant bit of each pixel, are flawed. Aletheia implements two attacks to these methods: the Sample Pairs Analysis (SPA) and the RS attack.

To execute the SPA attack to an included image with LSB replacement data hidden, use the following command:

```bash
$./aletheia.py spa sample_images/lena_lsbr.png 
Hiden data found in channel R 0.0930809062336
Hiden data found in channel G 0.0923858529528
Hiden data found in channel B 0.115466382367
```

The command used to perform the RS attack is similar:

```bash
$./aletheia.py rs sample_images/lena_lsbr.png 
Hiden data found in channel R 0.215602586771
Hiden data found in channel G 0.210351910548
Hiden data found in channel B 0.217878287806
```

In both cases the results provides an estimation of the embedding rate. 



