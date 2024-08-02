clear all;
load("kkk.mat");
rng(1);%设置随机种子
cc1=3.7417749e4; %W·m^2
cc2=1.438769e4; % um·K

lambda=[3.62 3.7 4.03 4.19 4.38 4.63];

t_t = [];
for i = 300 : 20 : 1200
    t_t = [t_t, i];    
end
t_k = t_t + 273.15;

data_E = [];
data_K = [];
data0 = [];
for TK = t_k
    Mb = cc1 ./(lambda.^5)./(exp(cc2 ./(lambda*TK))-1);
    T_gray = Mb .* kkk;
    data0 = [data0;T_gray];
    for j = 1 : 10
        Gaussnoise = normrnd(T_gray, (0.05*T_gray));

        data_E = [data_E;Gaussnoise];
        data_K = [data_K;TK];
    end

end

%发射率1 下降
simE1 = [0.8, 0.71, 0.48, 0.39, 0.27, 0.15];
%发射率1 上升
simE2 = [0.35, 0.385, 0.538,0.648, 0.75, 0.91];
% 先上升后下降
simE3 = [0.48,0.655, 0.85, 0.825, 0.685, 0.45];
% 线下降后上升
simE4 = [0.847, 0.712, 0.514, 0.536, 0.694, 0.856];

dataoutput1 = simE1.* data_E;
dataoutput2 = simE2.* data_E;
dataoutput3 = simE3.* data_E;
dataoutput4 = simE4.* data_E;
% dataoutput1 =  data_E ./ simE1;
% dataoutput2 =  data_E ./ simE2;
% dataoutput3 =  data_E ./ simE3;
% dataoutput4 = data_E ./ simE4;


% dataoutput1 = cat(2,dataoutput1,data_K);
% dataoutput2 = cat(2,dataoutput2,data_K);
% dataoutput3 = cat(2,dataoutput3,data_K);
% dataoutput4 = cat(2,dataoutput4,data_K);

% data_output = data_E;
% data_output_K = data_K;
data_output = cat(1,dataoutput1, dataoutput2,dataoutput3,dataoutput4);
data_output_K = cat(1,data_K, data_K, data_K, data_K);

num_tu = size(data_output,1);
tu = zeros(6,6,num_tu);
for k = 1 : num_tu
    for i = 1 : 6
        for j = 6: -1: 1
            tu(i,j,k) = abs(data_output(k,i) - data_output(k,j)) / (data_output(k,i) + data_output(k,j)) * 255;
        end
    end
end
image(uint8(tu(:,:,1)));



% data_output = cat(2, data_output, data_output_K);

% save data_fangzhen229 data_output

save data_tu_0229 tu data_output_K













