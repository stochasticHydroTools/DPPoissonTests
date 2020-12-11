#Modify any parameters down below and run this script to generate the g(r) of a distribution of +/- charges in a triply periodic environment.
#This script will output the average of a couple of identical simulations. In addition, this script will also output results from a DHO theory using the same potential as the simulations and a hard sphere one.

r_ion=2e-10 #Longitude units
C=1.6e-19 #Charge units
kT=4.11e-21 #Energy units
vacuumPermittivity=8.8541878e-12

#These two sets of potential parameters are the ones used in the other tests
#p=6
#gw=0.25
#r_m=1.5
#U0=0.700471

p=2
U0=0.2232754
gw=0.02
r_m=1

Molarity=0.05 #In mol/L. A low molarity will result in a better match with the DHO theory
N=8192 #Number of particles, it will only affect performance and statistics, you can use whatever number of particles.
waterPermittivity=78.5

#Times in diffusive times
simulationTime=500
printTime=1
listOfTimeSteps="1e-2 2e-3"

numberSimulations=2 #Number of identical simulations that will be ran for averaging

L=$(echo $N $Molarity $r_ion | awk '{print (0.5*$1/(6.022e23*$2)*0.001)^(1/3.0)/$3}')
T=1.0
rh=1
vis=$(echo 1 | awk '{pi=atan2(0,-1); printf("%.14g\n", 1/(6.0*pi))}')
D0=$(echo $T $vis $rh | awk '{pi=atan2(0,-1);printf("%.14g\n",$1/(6*pi*$2*$3))}')
tauD0=$(echo $D0 $rh | awk '{print $2*$2/$1}')
sigma=$(echo $rh | awk '{print 2*$1}')
permitivity=$(echo $waterPermittivity | awk '{printf "%.14g\n", $1*'$vacuumPermittivity'*'$kT'*'$r_ion'/('$C'**2)}')

if ! command -v datamash &> /dev/null
then
    echo "WARNING: I use datamash to average the output of several runs. I wont be able to do so without it."
fi
function createDataMain {

cat<<EOF > $1
Lxy $L
H $L
numberSteps $nsteps
printSteps $printSteps
relaxSteps $relaxSteps
dt         $dt
numberParticles $N
temperature    $T
viscosity   $vis
hydrodynamicRadius $rh
outfile /dev/stdout
readFile /dev/stdin
sigma   $sigma
U0 $U0
r_m $r_m
p $p
triplyPeriodic
noWall
tolerance 1e-4
permitivity $permitivity
gw $gw
split $split
EOF
}

function optimizeSplit {
    relaxSteps=100
    printSteps=-1
    nsteps=1000
    sp=0
    fpsPrev=0
    for split in $(seq 0.07 0.005 2)
    do
	createDataMain data.main
	fps=$(bash tools/init.sh data.main | ../poisson data.main 2>&1 | tee log | grep FPS | cut -f2 -d:)
	if $(echo $? | awk '{exit(!$1)}');
	then
	    echo "UAMMD failed" >/dev/stderr
	    tail -5 log >/dev/stderr
	    continue
	fi
	echo $(cat data.main | grep split) "FPS:" $fps >/dev/stderr
	if ! echo $fps $fpsPrev | awk '$1<$2{exit 1}'
	then
	    break
	fi
	sp=$split;
	fpsPrev=$fps
    done 
    echo $sp
}

function rdfTheory {
    debyeLength=$(echo 1 | awk '{print sqrt('$permitivity'/(2*'$Molarity'/0.001*'$r_ion'**3*6.022e23));}');
    dr=$(echo $1 $2 | awk '{print $1/$2}');
    g++ tools/dhordf.cpp -o tools/dhordf
    seq 0 $dr $1 | ./tools/dhordf $3 $U0 $permitivity $r_m $p $gw $debyeLength $sigma $4
}

mkdir -p tools
cd tools
if ! test -f rdf
then
    git clone https://github.com/raulppelaez/RadialDistributionFunction
    if ! (cd RadialDistributionFunction && mkdir build && cd build && cmake .. && make -j4)
    then
	echo "ERROR When compiling RadialDistributionFunction, check inside tools/ and ensure tools/rdf exists before continuing" > /dev/stderr
	exit 1
    fi
    cp RadialDistributionFunction/build/bin/rdf ..
    rm -rf RadialDistributionFunction
fi
cd ..

rdfTheory 15 2000 -1 1 > rdf.pm.hs.theo
rdfTheory 15 2000 -1 0 > rdf.pm.lj.theo
for dt in $listOfTimeSteps
do
    echo "Doing dt $dt"
    split=$(optimizeSplit)
    echo "Using split $split"
    relaxSteps=$(echo $dt $tauD0 | awk '{print int(10*$2/$1+0.5)}')
    nsteps=$(echo $dt $tauD0 $simulationTime | awk '{print int($2*$3/$1+0.5)}')
    printSteps=$(echo $dt $tauD0 $printTime | awk '{print int($2*$3/$1+0.5)}')
    datamain=data.main.dt$dt
    createDataMain $datamain
    NstepsInFile=$(echo $nsteps $printSteps | awk '{print int($1/$2)}')
    for i in $(seq 1 $numberSimulations)
    do
	bash tools/init.sh $datamain | ../poisson $datamain 2>log | # tee pos.dat | #uncomment to store positions to disk
	    awk '{print $1, $2, $3, ($4+1)*0.5}' | 
	    tools/rdf  -useTypes -N $N -Nsnapshots $NstepsInFile -nbins 300 -rcut 15 -L $L |
	    awk '/#/{f++}f==2&&!/#/' > rdf.$i.sim
    done
    cat rdf.*.sim | sort -g -k1 |
	datamash -W groupby 1 mean 2 sstdev 2 |
	sort -g -k1 | awk '{print $1, $2, $3/sqrt('$numberSimulations')}' > rdf.dt$dt.dat
    rm -f rdf.*.sim
done
mkdir -p results
mv data.main.* rdf.* log results
