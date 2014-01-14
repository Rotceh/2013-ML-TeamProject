# LaTeX on Mac

Install MacTeX <http://tug.org/mactex/>.

Install fonts in `requirement/latex_font.zip` by double clicking them.

## Compile TeX File

Through command line

```
mkdir build
xelatex -output-directory=build final_report.tex
open build/final_report.pdf
```

----

# LaTeX on Ubuntu

Install TexLive and a GUI editor TexMaker.

```
git clone https://github.com/scottkosty/install-tl-ubuntu
cd install-tl-ubuntu
git checkout devel
sudo ./install-tl-ubuntu
sudo apt-get install texmaker  # for latex editor
```

## Fonts

Install fonts in `requirement/latex_font.zip` by putting them
under a folder `/usr/share/fonts/opentype`. Then run

```bash
sudo fc-cache -vf
```

## Compile TeX File

Through command line

```
mkdir build
xelatex -output-directory=build final_report.tex
open build/final_report.pdf
```

## Modify TexMaker

Change pdflatex to xelatex.
