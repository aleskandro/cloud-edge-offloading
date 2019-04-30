param nbServiceProviders;
param nbServers;
param nbResources;
param nbOptions{i in 1..nbServiceProviders};
param bandwidthSaving{i in 1..nbServiceProviders, j in 1..nbOptions[i]};
param nbContainers{i in 1..nbServiceProviders, j in 1..nbOptions[i]};
param requiredResources{i in 1..nbServiceProviders, j in 1..nbOptions[i], k in 1..nbContainers[i,j], l in 1..nbResources};
param availableResources{i in 1..nbServers, j in 1..nbResources};

var option{i in 1..nbServiceProviders, j in 1..nbOptions[i]}, binary;
var placedContainer{i in 1..nbServiceProviders, j in 1..nbOptions[i], k in 1..nbContainers[i,j], l in 1..nbServers}, binary;



maximize opt: sum{i in 1..nbServiceProviders} sum{j in 1..nbOptions[i]} bandwidthSaving[i,j]*option[i,j];

s.t. ct1{i in 1..nbServiceProviders,
        j in 1..nbOptions[i],
        c in 1..nbContainers[i,j]}:
        sum{m in 1..nbServers} placedContainer[i,j,c,m] = option[i,j];

s.t. ct2{m in 1..nbServers, l in 1..nbResources}:
        sum{i in 1..nbServiceProviders}
            sum{j in 1..nbOptions[i]}
                sum{c in 1..nbContainers[i,j]}
                    placedContainer[i,j,c,m] * requiredResources[i,j,c,l] <= availableResources[m,l];

s.t. ct3{i in 1..nbServiceProviders}:
        sum{j in 1..nbOptions[i]}
            option[i,j] <= 1;

# Workaround to get totalResources
var totalResources{r in 1..nbResources};
s.t. ttRes1{r in 1..nbResources}: totalResources[r] = sum{s in 1..nbServers} availableResources[s,r];

var remainingResources{s in 1..nbServers, r in 1..nbResources};
s.t. rrRes1{s in 1..nbServers, r in 1..nbResources}:
        remainingResources[s,r] = availableResources[s,r] - 
            sum{i in 1..nbServiceProviders}
                sum{j in 1..nbOptions[i]}
                    sum{c in 1..nbContainers[i,j]}
                        placedContainer[i,j,c,s] * requiredResources[i,j,c,r];

var remainingTotalResources{r in 1..nbResources};
s.t. rrtRes1{r in 1..nbResources}:
        remainingTotalResources[r] = (sum{s in 1..nbServers} remainingResources[s,r])/ sum{s in 1..nbServers} availableResources[s,r];
solve;

display option;
display placedContainer;
display opt;

param avgOptions := (sum{i in 1..nbServiceProviders} nbOptions[i])/nbServiceProviders;
param avgContainers := (sum{i in 1..nbServiceProviders} ((sum {j in 1..nbOptions[i]} nbContainers[i,j])/nbOptions[i]))/nbServiceProviders;
display avgOptions;


printf "%f,%f,%f,%f\n", nbServers, avgContainers, avgOptions, opt >> "tresults/bandwidthByAvgOptions.csv";

display totalResources;
printf{r in 1..nbResources} "%f,", totalResources[r] >> "tresults/bandwidthByResources.csv";
printf "%f\n", opt >> "tresults/bandwidthByResources.csv";

display remainingTotalResources;

printf "%f,%f,%f", nbServers, avgContainers, avgOptions >> "tresults/remainingResourcesByAvgOptions.csv";
printf{r in 1..nbResources} ",%f", remainingTotalResources[r] >> "tresults/remainingResourcesByAvgOptions.csv";
printf "\n" >> "tresults/remainingResourcesByAvgOptions.csv";
