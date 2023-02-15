echo 'WHU-RS19' &&

str="A"
if [[ $3 =~ $str ]]
then
    echo 'x2'
    python testSCA.py --config ./configs/test/test-WHU-RS19sm-2.yaml --model $1 --gpu $2
else
    :
fi

str="B"
if [[ $3 =~ $str ]]
then
    echo 'x3'
    python testSCA.py --config ./configs/test/test-WHU-RS19sm-3.yaml --model $1 --gpu $2
else
    :
fi

str="C"
if [[ $3 =~ $str ]]
then
    echo 'x4'
    python testSCA.py --config ./configs/test/test-WHU-RS19sm-4.yaml --model $1 --gpu $2
else
    :
fi

str="D"
if [[ $3 =~ $str ]]
then
    echo 'x6'
    python testSCA.py --config ./configs/test/test-WHU-RS19sm-6.yaml --model $1 --gpu $2
else
    :
fi

str="E"
if [[ $3 =~ $str ]]
then
    echo 'x8'
    python testSCA.py --config ./configs/test/test-WHU-RS19sm-8.yaml --model $1 --gpu $2
else
    :
fi

str="F"
if [[ $3 =~ $str ]]
then
    echo 'x12'
    python testSCA.py --config ./configs/test/test-WHU-RS19sm-12.yaml --model $1 --gpu $2
else
    :
fi

str="G"
if [[ $3 =~ $str ]]
then
    echo 'x16'
    python testSCA.py --config ./configs/test/test-WHU-RS19sm-16.yaml --model $1 --gpu $2
else
    :
fi

str="H"
if [[ $3 =~ $str ]]
then
    echo 'x20'
    python testSCA.py --config ./configs/test/test-WHU-RS19sm-20.yaml --model $1 --gpu $2
else
    :
fi

str="I"
if [[ $3 =~ $str ]]
then
    echo 'x3.4'
    python testSCA.py --config ./configs/test/test-WHU-RS19sm-3.4.yaml --model $1 --gpu $2
else
    :
fi

str="J"
if [[ $3 =~ $str ]]
then
    echo 'x9.7'
    python testSCA.py --config ./configs/test/test-WHU-RS19sm-9.7.yaml --model $1 --gpu $2
else
    :
fi

str="K"
if [[ $3 =~ $str ]]
then
    echo 'x17.6'
    python testSCA.py --config ./configs/test/test-WHU-RS19sm-17.6.yaml --model $1 --gpu $2
else
    :
fi

true
