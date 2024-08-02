clear all;
load("kkk.mat");
rng(1);%设置随机种子
cc1=3.7417749e4; %W·m^2
cc2=1.438769e4; % um·K

% lambda=[3.62 3.7 4.03 4.19 4.38 4.63];%3.643  3.707 4.036 3.664 4.382 4.614
lambda=[3.75  3.95 4.15 4.35  4.55 4.75];
t_t = [];
for i = 300 : 1 : 1200
    t_t = [t_t, i];    
end
t_k = t_t + 273.15;
% t_k = [600 700 800];
data_E = [];
data_K = [];
data0 = [];
for TK = t_k
    Mb = cc1 ./(lambda.^5)./(exp(cc2 ./(lambda*TK))-1);
    T_gray = Mb .* kkk;
    Gaussnoise = normrnd(T_gray, (0.02*T_gray));
    data0 = [data0;Gaussnoise];
    % for j = 1 : 10
    %     Gaussnoise = normrnd(T_gray, (0.05*T_gray));
    % 
    %     data_E = [data_E;Gaussnoise];
    %     data_K = [data_K;TK];
    % end

end
% cankao = cat(2,data0,t_k');
% cankao = data0(2,:);
% cankao_k = t_k(2)';
data_E = data0;
% data_E(2,:) = [];
data_K = t_k';
% data_K(2) = [];
% %发射率1 下降
% simE1 = [0.8, 0.71, 0.48, 0.39, 0.27, 0.15];
%发射率1 下降
simE1 = [0.83, 0.72, 0.63, 0.56, 0.51, 0.48];
% figure
% plot(lambda,simE1);
% %发射率1 上升
% simE2 = [0.35, 0.385, 0.538,0.648, 0.75, 0.91];
%发射率1 上升
simE2 = [0.52, 0.55, 0.60,0.67, 0.76, 0.87];
% figure
% plot(lambda,simE2);
% 先上升后下降
simE3 = [0.55,0.71, 0.85, 0.83, 0.69, 0.54];
% figure
% plot(lambda,simE3);
% 线下降后上升
simE4 = [0.84, 0.68, 0.53, 0.51, 0.67, 0.85];
% figure
% plot(lambda,simE4);
%W型
simE5 = [0.83, 0.61, 0.85, 0.73, 0.55, 0.80];
% figure
% plot(lambda,simE5);
%M型
simE6 = [0.48, 0.79,0.65,0.50,0.82,0.60];

% figure
% plot(lambda,simE6);

% %发射率1 下降
% simE1 = [0.8, 0.71, 0.48, 0.39, 0.27, 0.15];
%发射率1 下降
simE7 = [0.68 0.76  0.70  0.62  0.54  0.46 ];
% figure
% plot(lambda,simE7);
% %发射率1 上升
% simE2 = [0.35, 0.385, 0.538,0.648, 0.75, 0.91];
%发射率1 上升
simE8 = [0.48 0.57 0.66 0.74 0.80 0.69];
% figure
% plot(lambda,simE8);
% 先上升后下降
simE9 = [0.65,0.55, 0.50, 0.53, 0.67, 0.83];
% figure
% plot(lambda,simE9);
% 线下降后上升
simE10 = [0.85, 0.68, 0.52, 0.48, 0.53, 0.65];
% figure
% plot(lambda,simE10);
%W型
simE11 = [0.84 0.66 0.54 0.66 0.84 0.55 ];
% figure
% plot(lambda,simE11);
%M型
simE12 = [0.55 0.74 0.58 0.48 0.58 0.75];

% figure
% plot(lambda,simE12);


dataoutput1 = simE1.* data_E;
dataoutput2 = simE2.* data_E;
dataoutput3 = simE3.* data_E;
dataoutput4 = simE4.* data_E;
dataoutput5 = simE5.* data_E;
dataoutput6 = simE6.* data_E;
dataoutput7 = simE7.* data_E;
dataoutput8 = simE8.* data_E;
dataoutput9 = simE9.* data_E;
dataoutput10 = simE10.* data_E;
dataoutput11 = simE11.* data_E;
dataoutput12 = simE12.* data_E;
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
data_output = cat(1,dataoutput1, dataoutput2,dataoutput3,dataoutput4,dataoutput5,dataoutput6,dataoutput7, dataoutput8,dataoutput9,dataoutput10,dataoutput11,dataoutput12);
data_output_K = cat(1,data_K, data_K, data_K, data_K, data_K, data_K,data_K, data_K, data_K, data_K, data_K, data_K);
EEE = cat(1,simE1,simE2,simE3,simE4,simE5,simE6,simE7,simE8,simE9,simE10,simE11,simE12);
% num_tu = size(data_output,1);
% tu = zeros(6,6,num_tu);
% for k = 1 : num_tu
%     for i = 1 : 6
%         for j = 6: -1: 1
%             % tu(i,6 - j + 1,k) = abs(data_output(k,i) - data_output(k,j)) / (data_output(k,i) + data_output(k,j)) * 255;
%             tu(i,j,k) = abs(data_output(k,i) - data_output(k,j)) / (data_output(k,i) + data_output(k,j)) * 255;
% 
%         end
%     end
%     % imwrite(tu(:,:,k),jet,strcat('仿真二维/',num2str(k),'.bmp'));
% end
% image(uint8(tu(:,:,501)));
% figure;
% imshow(uint8(tu(:,:,501)),Colormap= jet);



% data_output = cat(2, data_output, data_output_K);

% save data_chuangtong_GA_noise cankao cankao_k data_output data_output_K simE1 simE2 simE3 simE4 simE5 simE6

% save data_fangzhen229 data_output

% save data_tu_0229 tu data_output_K
save noise_data_wangluo_327_2 data_output data_output_K data0 data_K lambda cc1 cc2;


% ex_output = cat(1,dataoutput1(401,:), dataoutput2(401,:),dataoutput3(401,:),dataoutput4(401,:),dataoutput5(401,:),dataoutput6(401,:),dataoutput7(401,:), dataoutput8(401,:),dataoutput9(401,:),dataoutput10(401,:),dataoutput11(401,:),dataoutput12(401,:));









