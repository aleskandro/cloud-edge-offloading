/*********************************************
 * OPL 12.6.1.0 Model
 * Author: aleskandro
 * Creation Date: Feb 23, 2019 at 5:06:16 PM
 *********************************************/

int nbServiceProviders = ...; // Number of Service providers
int nbServers = ...; // Number of available servers
int nbResources = ...; // Number of resource categories per server
range serviceProviders = 1..nbServiceProviders;
range servers = 1..nbServers;
range resources = 1..nbResources;
int nbOptions[serviceProviders] = ...; // Number of options provided by each service provider
int maxOptions = max(sp in serviceProviders) nbOptions[sp];
range options = 1..maxOptions;
int bandwidthSaving[serviceProviders][options] = ...; // Bandwidth saving given the service provider and the option
int nbContainers[serviceProviders][options] = ...; // Number of containers per (option, service provider)
int maxContainers = max(sp in serviceProviders, opt in options) nbContainers[sp][opt];
range containers = 1..maxContainers;
int requiredResources[serviceProviders][options][containers][resources] = ...; // Required resources by the sp i, option j, container c, resource l
int availableResources[servers][resources] = ...; // Available resource l in the server m

dvar int option[serviceProviders][options] in 0..1;
dvar int containerPlaced[serviceProviders][options][containers][servers] in 0..1;

maximize
  sum(i in serviceProviders) sum(j in options : j <= nbOptions[i]) bandwidthSaving[i][j] * option[i][j];
  
subject to {
	forall (i in serviceProviders)
	  forall (j in options : j <= nbOptions[i])
	    forall(c in containers : c <= nbContainers[i][j])
	      ct1:
	      	sum (m in servers)
	      	  containerPlaced[i][j][c][m] == option[i][j];
	
	forall (m in servers)
	  forall (l in resources)
	    ct2:
	    	sum (i in serviceProviders)
	    	  sum (j in options : j <= nbOptions[i])
	    	    sum (c in containers : c <= nbContainers[i][j])
	    	      containerPlaced[i][j][c][m] * requiredResources[i][j][c][l] <= availableResources[m][l];
	
	forall (i in serviceProviders)
	  ct3:
	  	sum (j in options : j <= nbOptions[i])
	    	option[i][j] == 1;
	    
}
  



