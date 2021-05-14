
### Brute-force password attack

Aletheia can find the password used to hide a message using brute force from a list of passwords. 




#### Attack to OpenStego

First we are going to hide a message into a PNG image using OpenStego:

```bash
$ openstego embed -p 123456 -mf secret.txt -cf sample_images/lena.png -sf stego.png
```

Now we can use Aletheia to find the password:

```bash
./aletheia.py brute-force-openstego stego.png resources/passwords.txt 
Using 16 processes
Completed: 0.0%    
Password found: 123456
```





#### Attack to StegHide

First we are going to hide a message into a JPEG image using steghide:

```bash
$ steghide embed -cf cover.jpg -sf test.jpg -p 12345ab
Hello World!
embedding standard input in "cover.jpg"... done
writing stego file "image.jpg"... done
```

Now we can use Aletheia to find the password:

```bash
./aletheia.py brute-force-steghide test.jpg resources/passwords.txt 
Using 16 processes
Completed: 0.4%    
Password found: 12345ab
```


#### Attack to Outguess

First, we hide a file:

```bash
$ outguess -k maggie -d test.txt sample_images/alaska2jpg/01294.jpg test.jpg
Reading sample_images/alaska2jpg/01294.jpg....
JPEG compression quality set to 75
Extracting usable bits:   28144 bits
Correctable message size: 12922 bits, 45.91%
Encoded 'test.txt': 40 bits, 5 bytes
Finding best embedding...
    0:    34(47.2%)[85.0%], bias    39(1.15), saved:    -1, total:  0.12%
    2:    30(42.3%)[75.0%], bias    31(1.03), saved:    -1, total:  0.11%
    4:    27(37.5%)[67.5%], bias    30(1.11), saved:     0, total:  0.10%
    6:    30(42.3%)[75.0%], bias    25(0.83), saved:    -1, total:  0.11%
   23:    26(36.1%)[65.0%], bias    25(0.96), saved:     0, total:  0.09%
   45:    29(40.3%)[72.5%], bias    20(0.69), saved:    -1, total:  0.10%
45, 49: Embedding data: 40 in 28144
Bits embedded: 72, changed: 29(40.3%)[72.5%], bias: 20, tot: 27563, skip: 27491
Foiling statistics: corrections: 15, failed: 0, offset: -nan +- -nan
Total bits changed: 49 (change 29 + bias 20)
Storing bitmap into data...
Writing test.jpg....
```

Now we can use Aletheia to find the password:

```bash
./aletheia.py brute-force-outguess test2.jpg resources/passwords.txt                                                                                              
Using 16 processes                                                                                        
Candidate password: cocacola, filetype found: application/x-dosexec                                       
Candidate password: maggie, filetype found: text/plain                                                    
Candidate password: jermaine, filetype found: audio/x-mp4a-latm                                           
Candidate password: 20041986, filetype found: application/x-dosexec                                       
...

```

Outguess does not tell us if the password is correct so Aletheia needs to check
the file type to see if the password is a good candiate. In the example, the
second password is the right password. The other options are noise.



#### Attacks with a generic command

```bash
$ ./aletheia.py brute-force-generic
./aletheia.py brute-force-generic <unhide command> <passw file>

Example:
./aletheia.py brute-force-generic 'steghide extract -sf image.jpg -xf output.txt -p <PASSWORD> -f' resources/passwords.txt

```

We need to provide to Aletheia the command used to extract the secret message. This command needs a "&lt;PASSWORD&gt;" section that will be replaced by every password to try. The second parameter to provide to Aletheia is the list of passwords to try. Aletheia comes with a file that contains 1000000 passwords.

Let's use Aletheia to crack Steghide. 


First we are going to hide a message into a JPEG image using steghide:

```bash
$ steghide embed -cf cover.jpg -sf image.jpg -p 12345ab
Hello World!
embedding standard input in "cover.jpg"... done
writing stego file "image.jpg"... done
```

Now, we can find the password using the following command:

```bash
$ ./aletheia.py bf-generic 'steghide extract -sf image.jpg -xf output.txt -p <PASSWORD> -f' resources/passwords.txt
Using 16 processes
Completed: 4.5%    
Password found: 12345ab
```

The secret message was:

```bash
$ cat output.txt
Hello World!
```



