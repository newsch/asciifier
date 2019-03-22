# asciifier

Artisanal ascii images

## Scripts

There are currently three Python scripts involved here, each with commandline interfaces.

- `asciifier.py` does most of the work by taking an input image, resizing it, and converting it to text. 
- `pagifier.py` splits text into multiple columns, which I used for making  pictures span across multiple pages.
- `layerifier.py` is used for doing multiple passes of printing on a line-by-line basis. My printer treats `\r` and `\n` differently, splitting the literal "carriage return" and "line feed" into separate steps, so you can print over the same line multiple times. This script takes multiple files in and splices their lines together. I've used this to get darker results from faded ink cartridges.
