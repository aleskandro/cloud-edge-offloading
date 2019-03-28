1;
function bwopt()
    fid = fopen('results/bandwidthByAvgOptions.csv', 'r');
    bandwidthByAvgOptions = cell2mat(textscan(fid, '%f %f', 'delimiter', ','));
    bandwidthByAvgOptions = sortrows(bandwidthByAvgOptions, [1]);
    plot(bandwidthByAvgOptions(:,[1]), bandwidthByAvgOptions(:,[2]), "linewidth", 2);
    set(gca, "linewidth", 2, "fontsize", 12)
    xlim([1 max(bandwidthByAvgOptions(:,[1]))]);
endfunction;

function rropt()
    fid = fopen('results/remainingResourcesByAvgOptions.csv', 'r');
    rrByAvgOpt = cell2mat(textscan(fid, '%f %f %f', 'delimiter', ','));
    rrByAvgOpt = sortrows(rrByAvgOpt, [1]);
    plot(rrByAvgOpt(:,[1]), rrByAvgOpt(:,[2,3]), "linewidth", 2);
    set(gca, "linewidth", 2, "fontsize", 12)
    xlim([1 max(rrByAvgOpt(:,[1]))]);

endfunction;

function graphs()
    fid = fopen('results/bandwidthByAvgOptions.csv', 'r');
    bandwidthByAvgOptions = cell2mat(textscan(fid, '%f %f', 'delimiter', ','));
    bandwidthByAvgOptions = sortrows(bandwidthByAvgOptions, [1]);

    fid = fopen('results/remainingResourcesByAvgOptions.csv', 'r');
    rrByAvgOpt = cell2mat(textscan(fid, '%f %f %f', 'delimiter', ','));
    rrByAvgOpt = sortrows(rrByAvgOpt, [1]);
   
    subplot(2,1,1)
    plot(rrByAvgOpt(:,[1]), rrByAvgOpt(:,[2,3]), "linewidth", 2);
    set(gca, "linewidth", 2, "fontsize", 24)
    xlim([1 max(rrByAvgOpt(:,[1]))]);
    ylim([0 1]);
    subplot(2,1,2)
    plot(bandwidthByAvgOptions(:,[1]), bandwidthByAvgOptions(:,[2]), "linewidth", 2);
    set(gca, "linewidth", 2, "fontsize", 24)
    xlim([1 max(bandwidthByAvgOptions(:,[1]))]);
    ylim([0 ceil(max(bandwidthByAvgOptions(:, [2])))]);

endfunction;


