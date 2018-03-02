#! /bin/bash

#string="hello-world"
#prefix="hell"
#suffix="ld"
#foo=${string#$prefix}
#foo=${foo%$suffix}

#for i in `ls ../../CPI/*_01_14*100 -d` ; do ./make_pdf.sh $i; done

for i in `ls $1/*.png`
  do
  string=$i
  prefix=$1'/'
  suffix="png"
  foo=${string#$prefix}
  foo=${foo%$suffix}
  echo $i
  convert $i /tmp/${foo}pdf
done

gs -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -sOutputFile=${1}.pdf /tmp/*.pdf

for i in `ls $1/*.png`
  do
  string=$i
  prefix=$1'/'
  suffix="png"
  foo=${string#$prefix}
  foo=${foo%$suffix}
  echo $i
  rm /tmp/${foo}pdf
done


