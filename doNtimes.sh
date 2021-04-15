if [ $# -ne 3 ]; then
  echo "Esegui N volte 2 eseguibili"
  echo "Utilizzo : $0 ./eseguibile1 ./eseguibile2 NumeroDiRipertizioni"
  echo "NB: ricordarsi di eslpicitare il PATH agli eseguibili"
  exit
fi

EXE1=$1
EXE2=$2
LOOP=$3

rm simul.dat fit.dat

for i in `seq 1 $LOOP`; do
  $EXE1 1 100 $i
  $EXE2
if [[ $(($i % 100)) -eq 0 ]]
  then
    echo $i
  fi
done

echo "Ho eseguito " $LOOP " volte " $EXE1 " e " $EXE2
