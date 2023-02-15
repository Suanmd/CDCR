echo 'RSC11' &&

str="A"
if [[ $3 =~ $str ]]
then
    echo 'x2'
    python test.py --config ./configs/test/test-RSC11sm-2.yaml --model $1 --gpu $2
else
    :
fi

str="B"
if [[ $3 =~ $str ]]
then
    echo 'x3'
    python test.py --config ./configs/test/test-RSC11sm-3.yaml --model $1 --gpu $2
else
    :
fi

str="C"
if [[ $3 =~ $str ]]
then
    echo 'x4'
    python test.py --config ./configs/test/test-RSC11sm-4.yaml --model $1 --gpu $2
else
    :
fi

str="D"
if [[ $3 =~ $str ]]
then
    echo 'x6'
    python test.py --config ./configs/test/test-RSC11sm-6.yaml --model $1 --gpu $2
else
    :
fi

str="E"
if [[ $3 =~ $str ]]
then
    echo 'x8'
    python test.py --config ./configs/test/test-RSC11sm-8.yaml --model $1 --gpu $2
else
    :
fi

str="F"
if [[ $3 =~ $str ]]
then
    echo 'x12'
    python test.py --config ./configs/test/test-RSC11sm-12.yaml --model $1 --gpu $2
else
    :
fi

str="G"
if [[ $3 =~ $str ]]
then
    echo 'x16'
    python test.py --config ./configs/test/test-RSC11sm-16.yaml --model $1 --gpu $2
else
    :
fi

str="H"
if [[ $3 =~ $str ]]
then
    echo 'x20'
    python test.py --config ./configs/test/test-RSC11sm-20.yaml --model $1 --gpu $2
else
    :
fi

str="I"
if [[ $3 =~ $str ]]
then
    echo 'x3.4'
    python test.py --config ./configs/test/test-RSC11sm-3.4.yaml --model $1 --gpu $2
else
    :
fi

str="J"
if [[ $3 =~ $str ]]
then
    echo 'x9.7'
    python test.py --config ./configs/test/test-RSC11sm-9.7.yaml --model $1 --gpu $2
else
    :
fi

str="K"
if [[ $3 =~ $str ]]
then
    echo 'x17.6'
    python test.py --config ./configs/test/test-RSC11sm-17.6.yaml --model $1 --gpu $2
else
    :
fi

true
