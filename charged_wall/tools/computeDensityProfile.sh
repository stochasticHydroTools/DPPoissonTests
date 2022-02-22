positions=$1
datamain="data.main"
K=$2
nbins=100

nsteps=$(cat $positions | grep -c "#");
Hhalf=$(grep -Eo '^H[[:space:]].*' $datamain | awk '{print $2*0.5}')
H=$(grep -Eo '^H[[:space:]].*' $datamain | awk '{print $2}')
L=$(grep -Eo '^Lxy[[:space:]].*' $datamain | awk '{print $2}')
N=$(grep -Eo '^numberParticles[[:space:]].*' $datamain | awk '{print $2}')
ep=$(grep -Eo '^permitivity[[:space:]].*' $datamain | awk '{print $2}')
nm=$(echo 1 | awk '{printf "%.14g\n", 2*'$K'^2*'$ep'/('$H'-2)^2;}')

function histogram {
nbins=$1
upper=$2
lower=$3
awk '{min='$lower'; max='$upper'; nbins='$nbins';b=int(($1-min)/(max-min)*nbins); h[b] += $2;}
     END{for(i=0;i<nbins;i++){
	   z=(i+0.5)/nbins*(max-min)+min;
	   print z, h[i]*1.0;
	 }
	}'
}

#Normalize the histogram to a concentration, so that int_0^H c(z)dz = 1
histoToConcentration=$(echo 1 | awk '{print '$nbins'/('$N'*'$nsteps'*'$H')}')
#Normalize the concentration into as number density, so that int_0^H n(z)dz = N/(L^2*H)
concentrationToNumberDensity=$(echo 1 | awk '{print '$N'/'$L'^2}')
norm=$(echo 1 | awk '{print 1.0/('$concentrationToNumberDensity'*'$histoToConcentration')}')

cat $positions |
    grep -v "#" |
    awk '{printf "%.14g %.14g\n", sqrt($3*$3),1.0/('$norm'*'$nm')}' |
    histogram $nbins $Hhalf 0
