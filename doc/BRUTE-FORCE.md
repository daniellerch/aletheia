
### Brute-force password attack

Aletheia can find the password used to hide a message using brute force from a list of passwords. 


```bash
$ ./aletheia.py brute-force
./aletheia.py brute-force <unhide command> <passw file>

Example:
./aletheia.py brute-force 'steghide extract -sf image.jpg -xf output.txt -p <PASSWORD> -f' resources/passwords.txt

```

We need to provide to Aletheia the command used to extract the secret message. This command needs a "&lt;PASSWORD&gt;" section that will be replaced by every password to try. The second parameter to provide to Aletheia is the list of passwords to try. Aletheia comes with a file that contains 1000000 passwords.

Let's use Aletheia to crack the known steganography tool Steghide. First we are going to hide a message into a JPEG image using steghide:


```bash
$ steghide embed -cf cover.jpg -sf image.jpg -p 12345ab
Hello World!
embedding standard input in "cover.jpg"... done
writing stego file "image.jpg"... done
```

Now, we can find the password using the following command:

```bash
$ ./aletheia.py brute-force 'steghide extract -sf image.jpg -xf output.txt -p <PASSWORD> -f' resources/passwords.txt
Using 16 processes
Completed: 4.5%    
Password found: 12345ab
```

The secret message was:

```bash
$ cat output.txt
Hello World!
```


