% clear;
% load('A_BP.mat');
% error_B = error_BP;
% load('A_GRNN.mat')
% error_GRNN = error_BP;
% load('A_RBF.mat')
% error_RBF = error_BP;
% load('A_resCNN.mat')
% error_resCNN = error_BP;
% load('A_Me.mat')
% error_Me = error_BP;
% 
% N = size(error_BP,2);
% X = 1:N;
% Y = [error_B;error_GRNN;error_RBF;error_resCNN;error_Me];
% figure
% % plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% scatter(X, error_B,65, 'ro');
% hold on;
% scatter(X, error_GRNN, 65,'g+');
% hold on;
% scatter(X,error_RBF,65,'b*');
% hold on;
% scatter(X,error_resCNN,65,'magentax');
% hold on;
% scatter(X,error_Me,65,'k^');
% % scatter(X, error_B, 'ro', X, error_GRNN, 'g+',X,error_RBF,'b*',X,error_resCNN,'c.',X,error_Me,'k^');
% legend('BP误差绝对值','GRNN误差绝对值','RBF误差绝对值','resCNN误差绝对值','本章算法误差绝对值','FontSize',22);
% % legend('GRNN误差绝对值');
% % legend('RBF误差绝对值');
% % legend('resCNN误差绝对值');
% % legend('本章算法误差绝对值');
% xlabel('测试集样本编号','FontSize',22);
% ylabel('误差绝对值 / K','FontSize',22);
% % string = {'BP测试集样本误差绝对值'};
% % title(string);
% set(gca,'FontSize',22);
% 
% xlim([1, N]);
% grid;

% 
% clear;
% load('D_BP.mat');
% error_B = error_BP;
% load('D_GRNN.mat')
% error_GRNN = error_BP;
% load('D_RBF.mat')
% error_RBF = error_BP;
% load('D_resCNN.mat')
% error_resCNN = error_BP;
% load('D_Me.mat')
% error_Me = error_BP;
% 
% N = size(error_BP,2);
% X = 1:N;
% Y = [error_B;error_GRNN;error_RBF;error_resCNN;error_Me];
% figure
% % plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% scatter(X, error_B,65, 'ro');
% hold on;
% scatter(X, error_GRNN, 65,'g+');
% hold on;
% scatter(X,error_RBF,65,'b*');
% hold on;
% scatter(X,error_resCNN,65,'magentax');
% hold on;
% scatter(X,error_Me,65,'k^');
% % scatter(X, error_B, 'ro', X, error_GRNN, 'g+',X,error_RBF,'b*',X,error_resCNN,'c.',X,error_Me,'k^');
% legend('BP误差绝对值','GRNN误差绝对值','RBF误差绝对值','resCNN误差绝对值','本章算法误差绝对值','FontSize',22);
% % legend('GRNN误差绝对值');
% % legend('RBF误差绝对值');
% % legend('resCNN误差绝对值');
% % legend('本章算法误差绝对值');
% xlabel('测试集样本编号','FontSize',22);
% ylabel('误差绝对值 / K','FontSize',22);
% % string = {'BP测试集样本误差绝对值'};
% % title(string);
% set(gca,'FontSize',22);
% 
% xlim([1, N]);
% grid;

clear;
load('D_BP.mat');
error_B = error_BP;

load('D_GRNN.mat')
error_GRNN = error_BP;
load('D_RBF.mat')
error_RBF = error_BP;
load('D_resCNN.mat')
error_resCNN = error_BP;
load('D_Me.mat')
error_Me = error_BP;

% N = size(error_BP,2);
% X = 1:N;
% Y = [error_B;error_GRNN;error_RBF;error_resCNN;error_Me];

a=error_B(1: 90);
n = randperm(90);
a = a(n);
b = error_GRNN(1: 90);
n = randperm(90);
b = b(n);
c=error_RBF(1: 90);
n = randperm(90);
c = c(n);
d=error_resCNN(1: 90);
n = randperm(90);
d = d(n);
e=error_Me(1: 90);
n = randperm(90);
e = e(n);
figure
set(gcf,'Position',[100 100 800 600]);
plot(1: 90, a, 'r-o', 1: 90, b, 'g-+',1:90,c,'b-*',1:90,d,'c-.',1:90,e,'k-^','LineWidth',1);
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% scatter(X, error_B, 'ro', X, error_GRNN, 'g+',X,error_RBF,'b*',X,error_resCNN,'c.',X,error_Me,'k^');
legend('BP','GRNN','RBF','ResCNN','Ours','FontSize',14,'location','northeast');
% legend('GRNN误差绝对值');
% legend('RBF误差绝对值');
% legend('resCNN误差绝对值');
% legend('本章算法误差绝对值');
xlabel('Sample number','FontSize',14);
ylabel('Absolute error / K','FontSize',14);
string = {'Class A'};
title(string);
set(gca,'FontSize',14);

xlim([1, 90]);
grid;
% disp(['amax：', num2str(max(a))])
% disp(['amin：', num2str(min(a))])
% disp(['bmax：', num2str(max(b))])
% disp(['bmin：', num2str(min(b))])
% disp(['cmax：', num2str(max(c))])
% disp(['cmin：', num2str(min(c))])
% disp(['dmax：', num2str(max(d))])
% disp(['dmin：', num2str(min(d))])
% disp(['emax：', num2str(max(e))])
% disp(['emin：', num2str(min(e))])
res = [max(a),min(a),max(b),min(b),max(c),min(c),max(d),min(d),max(e),min(e)];
% disp([num2str(max(b))])
% disp([num2str(min(b))])
% disp([num2str(max(c))])
% disp([ num2str(min(c))])
% disp([ num2str(max(d))])
% disp([num2str(min(d))])
% disp([num2str(max(e))])
% disp([num2str(min(e))])
clear;
load('D_BP.mat');
error_B = error_BP;

load('D_GRNN.mat')
error_GRNN = error_BP;
load('D_RBF.mat')
error_RBF = error_BP;
load('D_resCNN.mat')
error_resCNN = error_BP;
load('D_Me.mat')
error_Me = error_BP;

a=error_B(91: 180);
n = randperm(90);
a = a(n);
b = error_GRNN(91: 180);
n = randperm(90);
b = b(n);
c=error_RBF(91: 180);
n = randperm(90);
c = c(n);
d=error_resCNN(91: 180);
n = randperm(90);
d = d(n);
e=error_Me(91: 180);
n = randperm(90);
e = e(n);
figure
set(gcf,'Position',[100 100 800 600]);
plot(1: 90, a, 'r-o', 1: 90, b, 'g-+',1: 90,c,'b-*',1: 90,d,'c-.',1: 90,e,'k-^','LineWidth',1);
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% scatter(X, error_B, 'ro', X, error_GRNN, 'g+',X,error_RBF,'b*',X,error_resCNN,'c.',X,error_Me,'k^');
legend('BP','GRNN','RBF','ResCNN','Ours','FontSize',14,'location','northeast');
% legend('GRNN误差绝对值');
% legend('RBF误差绝对值');
% legend('resCNN误差绝对值');
% legend('本章算法误差绝对值');
xlabel('Sample number','FontSize',14);
ylabel('Absolute error / K','FontSize',14);
string = {'Class B'};
title(string);
set(gca,'FontSize',14);


xlim([1, 90]);
grid;
clear;
load('D_BP.mat');
error_B = error_BP;

load('D_GRNN.mat')
error_GRNN = error_BP;
load('D_RBF.mat')
error_RBF = error_BP;
load('D_resCNN.mat')
error_resCNN = error_BP;
load('D_Me.mat')
error_Me = error_BP;

a=error_B(181: 270);
n = randperm(90);
a = a(n);
b = error_GRNN(181: 270);
n = randperm(90);
b = b(n);
c=error_RBF(181: 270);
n = randperm(90);
c = c(n);
d=error_resCNN(181: 270);
n = randperm(90);
d = d(n);
e=error_Me(181: 270);
n = randperm(90);
e = e(n);
figure
set(gcf,'Position',[100 100 800 600]);
plot(1: 90, a, 'r-o', 1: 90, b, 'g-+',1: 90,c,'b-*',1: 90,d,'c-.',1: 90,e,'k-^','LineWidth',1);
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% scatter(X, error_B, 'ro', X, error_GRNN, 'g+',X,error_RBF,'b*',X,error_resCNN,'c.',X,error_Me,'k^');
legend('BP','GRNN','RBF','ResCNN','Ours','FontSize',14,'location','northeast');
% legend('GRNN误差绝对值');
% legend('RBF误差绝对值');
% legend('resCNN误差绝对值');
% legend('本章算法误差绝对值');
xlabel('Sample number','FontSize',14);
ylabel('Absolute error / K','FontSize',14);
string = {'Class C'};
title(string);
set(gca,'FontSize',14);


xlim([1, 90]);
grid;
a=error_B(271: 360);
n = randperm(90);
a = a(n);
b = error_GRNN(271: 360);
n = randperm(90);
b = b(n);
c=error_RBF(271: 360);
n = randperm(90);
c = c(n);
d=error_resCNN(271: 360);
n = randperm(90);
d = d(n);
e=error_Me(271: 360);
n = randperm(90);
e = e(n);
figure
set(gcf,'Position',[100 100 800 600]);
plot(1: 90, a, 'r-o', 1: 90, b, 'g-+',1: 90,c,'b-*',1: 90,d,'c-.',1: 90,e,'k-^','LineWidth',1);
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% scatter(X, error_B, 'ro', X, error_GRNN, 'g+',X,error_RBF,'b*',X,error_resCNN,'c.',X,error_Me,'k^');
legend('BP','GRNN','RBF','ResCNN','Ours','FontSize',14,'location','northeast');
% legend('GRNN误差绝对值');
% legend('RBF误差绝对值');
% legend('resCNN误差绝对值');
% legend('本章算法误差绝对值');
xlabel('Sample number','FontSize',14);
ylabel('Absolute error / K','FontSize',14);
string = {'Class D'};
title(string);
set(gca,'FontSize',14);


xlim([1, 90]);
grid;

a=error_B(361: 450);
n = randperm(90);
a = a(n);
b = error_GRNN(361: 450);
n = randperm(90);
b = b(n);
c=error_RBF(361: 450);
n = randperm(90);
c = c(n);
d=error_resCNN(361: 450);
n = randperm(90);
d = d(n);
e=error_Me(361: 450);
n = randperm(90);
e = e(n);
figure
set(gcf,'Position',[100 100 800 600]);
plot(1: 90, a, 'r-o', 1: 90, b, 'g-+',1: 90,c,'b-*',1: 90,d,'c-.',1: 90,e,'k-^','LineWidth',1);
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% scatter(X, error_B, 'ro', X, error_GRNN, 'g+',X,error_RBF,'b*',X,error_resCNN,'c.',X,error_Me,'k^');
legend('BP','GRNN','RBF','ResCNN','Ours','FontSize',14,'location','northeast');
% legend('GRNN误差绝对值');
% legend('RBF误差绝对值');
% legend('resCNN误差绝对值');
% legend('本章算法误差绝对值');
xlabel('Sample number','FontSize',14);
ylabel('Absolute error / K','FontSize',14);
string = {'Class E'};
title(string);
set(gca,'FontSize',14);


xlim([1, 90]);
grid;
a=error_B(451: 540);
n = randperm(90);
a = a(n);
b = error_GRNN(451: 540);
n = randperm(90);
b = b(n);
c=error_RBF(451: 540);
n = randperm(90);
c = c(n);
d=error_resCNN(451: 540);
n = randperm(90);
d = d(n);
e=error_Me(451: 540);
n = randperm(90);
e = e(n);
figure
set(gcf,'Position',[100 100 800 600]);
plot(1: 90, a, 'r-o', 1: 90, b, 'g-+',1: 90,c,'b-*',1: 90,d,'c-.',1: 90,e,'k-^','LineWidth',1);
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% scatter(X, error_B, 'ro', X, error_GRNN, 'g+',X,error_RBF,'b*',X,error_resCNN,'c.',X,error_Me,'k^');
legend('BP','GRNN','RBF','ResCNN','Ours','FontSize',14,'location','northeast');
% legend('GRNN误差绝对值');
% legend('RBF误差绝对值');
% legend('resCNN误差绝对值');
% legend('本章算法误差绝对值');
xlabel('Sample number','FontSize',14);
ylabel('Absolute error / K','FontSize',14);
string = {'Class F'};
title(string);
set(gca,'FontSize',14);

xlim([1, 90]);
grid;



a=error_B(541: 630);
n = randperm(90);
a = a(n);
b = error_GRNN(541: 630);
n = randperm(90);
b = b(n);
c=error_RBF(541: 630);
n = randperm(90);
c = c(n);
d=error_resCNN(541: 630);
n = randperm(90);
d = d(n);
e=error_Me(541: 630);
n = randperm(90);
e = e(n);
figure
set(gcf,'Position',[100 100 800 600]);
plot(1: 90, a, 'r-o', 1: 90, b, 'g-+',1: 90,c,'b-*',1: 90,d,'c-.',1: 90,e,'k-^','LineWidth',1);
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% scatter(X, error_B, 'ro', X, error_GRNN, 'g+',X,error_RBF,'b*',X,error_resCNN,'c.',X,error_Me,'k^');
legend('BP','GRNN','RBF','ResCNN','Ours','FontSize',14,'location','northeast');
% legend('GRNN误差绝对值');
% legend('RBF误差绝对值');
% legend('resCNN误差绝对值');
% legend('本章算法误差绝对值');
xlabel('Sample number','FontSize',14);
ylabel('Absolute error / K','FontSize',14);
string = {'Class G'};
title(string);
set(gca,'FontSize',14);


xlim([1, 90]);
grid;
a=error_B(631: 720);
n = randperm(90);
a = a(n);
b = error_GRNN(631: 720);
n = randperm(90);
b = b(n);
c=error_RBF(631: 720);
n = randperm(90);
c = c(n);
d=error_resCNN(631: 720);
n = randperm(90);
d = d(n);
e=error_Me(631: 720);
n = randperm(90);
e = e(n);
figure
set(gcf,'Position',[100 100 800 600]);
plot(1: 90, a, 'r-o', 1: 90, b, 'g-+',1: 90,c,'b-*',1: 90,d,'c-.',1: 90,e,'k-^','LineWidth',1);
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% scatter(X, error_B, 'ro', X, error_GRNN, 'g+',X,error_RBF,'b*',X,error_resCNN,'c.',X,error_Me,'k^');
legend('BP','GRNN','RBF','ResCNN','Ours','FontSize',14,'location','northeast');
% legend('GRNN误差绝对值');
% legend('RBF误差绝对值');
% legend('resCNN误差绝对值');
% legend('本章算法误差绝对值');
xlabel('Sample number','FontSize',14);
ylabel('Absolute error / K','FontSize',14);
string = {'Class H'};
title(string);
set(gca,'FontSize',14);


xlim([1, 90]);
grid;
a=error_B(721: 810);
n = randperm(90);
a = a(n);
b = error_GRNN(721: 810);
n = randperm(90);
b = b(n);
c=error_RBF(721: 810);
n = randperm(90);
c = c(n);
d=error_resCNN(721: 810);
n = randperm(90);
d = d(n);
e=error_Me(721: 810);
n = randperm(90);
e = e(n);
figure
set(gcf,'Position',[100 100 800 600]);
plot(1: 90, a, 'r-o', 1: 90, b, 'g-+',1: 90,c,'b-*',1: 90,d,'c-.',1: 90,e,'k-^','LineWidth',1);
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% scatter(X, error_B, 'ro', X, error_GRNN, 'g+',X,error_RBF,'b*',X,error_resCNN,'c.',X,error_Me,'k^');
legend('BP','GRNN','RBF','ResCNN','Ours','FontSize',14,'location','northeast');
% legend('GRNN误差绝对值');
% legend('RBF误差绝对值');
% legend('resCNN误差绝对值');
% legend('本章算法误差绝对值');
xlabel('Sample number','FontSize',14);
ylabel('Absolute error / K','FontSize',14);
string = {'Class I'};
title(string);
set(gca,'FontSize',14);


xlim([1, 90]);
grid;
a=error_B(811: 900);
n = randperm(90);
a = a(n);
b = error_GRNN(811: 900);
n = randperm(90);
b = b(n);
c=error_RBF(811: 900);
n = randperm(90);
c = c(n);
d=error_resCNN(811: 900);
n = randperm(90);
d = d(n);
e=error_Me(811: 900);
n = randperm(90);
e = e(n);
figure
set(gcf,'Position',[100 100 800 600]);
plot(1: 90, a, 'r-o', 1: 90, b, 'g-+',1: 90,c,'b-*',1: 90,d,'c-.',1: 90,e,'k-^','LineWidth',1);
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% scatter(X, error_B, 'ro', X, error_GRNN, 'g+',X,error_RBF,'b*',X,error_resCNN,'c.',X,error_Me,'k^');
legend('BP','GRNN','RBF','ResCNN','Ours','FontSize',14,'location','northeast');
% legend('GRNN误差绝对值');
% legend('RBF误差绝对值');
% legend('resCNN误差绝对值');
% legend('本章算法误差绝对值');
xlabel('Sample number','FontSize',14);
ylabel('Absolute error / K','FontSize',14);
string = {'Class J'};
title(string);
set(gca,'FontSize',14);


xlim([1, 90]);
grid;



a=error_B(901: 990);
n = randperm(90);
a = a(n);
b = error_GRNN(901: 990);
n = randperm(90);
b = b(n);
c=error_RBF(901: 990);
n = randperm(90);
c = c(n);
d=error_resCNN(901: 990);
n = randperm(90);
d = d(n);
e=error_Me(901: 990);
n = randperm(90);
e = e(n);
figure
set(gcf,'Position',[100 100 800 600]);
plot(1: 90, a, 'r-o', 1: 90, b, 'g-+',1: 90,c,'b-*',1: 90,d,'c-.',1: 90,e,'k-^','LineWidth',1);
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% scatter(X, error_B, 'ro', X, error_GRNN, 'g+',X,error_RBF,'b*',X,error_resCNN,'c.',X,error_Me,'k^');
legend('BP','GRNN','RBF','ResCNN','Ours','FontSize',14,'location','northeast');
% legend('GRNN误差绝对值');
% legend('RBF误差绝对值');
% legend('resCNN误差绝对值');
% legend('本章算法误差绝对值');
xlabel('Sample number','FontSize',14);
ylabel('Absolute error / K','FontSize',14);
string = {'Class K'};
title(string);
set(gca,'FontSize',14);


xlim([1, 90]);
grid;

% a=error_B(901: 1080)+0.5*rand(1);
% n = randperm(90);
% a = a(n);
% b = error_GRNN(901: 1080)+0.5*rand(1);
% n = randperm(90);
% b = b(n);
% c=error_RBF(901: 1080)+0.5*rand(1);
% n = randperm(90);
% c = c(n);
% d=error_resCNN(901: 1080)+0.5*rand(1);
% n = randperm(90);
% d = d(n);
% e=error_Me(901: 1080)+0.5*rand(1);
% n = randperm(90);
% e = e(n);
a=error_B(991: 1080);
n = randperm(90);
a = a(n);
b = error_GRNN(991: 1080);
n = randperm(90);
b = b(n);
c=error_RBF(991: 1080);
n = randperm(90);
c = c(n);
d=error_resCNN(991: 1080);
n = randperm(90);
d = d(n);
e=error_Me(991: 1080);
n = randperm(90);
e = e(n);
figure
set(gcf,'Position',[100 100 800 600]);
plot(1: 90, a, 'r-o', 1: 90, b, 'g-+',1: 90,c,'b-*',1: 90,d,'c-.',1: 90,e,'k-^','LineWidth',1);
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% plot(1: N, error_B, 'r-o', 1: N, error_GRNN, 'g-+',1:N,error_RBF,'b-*',1:N,error_resCNN,'c-.',1:N,error_Me,'k-^');
% scatter(X, error_B, 'ro', X, error_GRNN, 'g+',X,error_RBF,'b*',X,error_resCNN,'c.',X,error_Me,'k^');
legend('BP','GRNN','RBF','ResCNN','Ours','FontSize',14,'location','northeast');
% legend('GRNN误差绝对值');
% legend('RBF误差绝对值');
% legend('resCNN误差绝对值');
% legend('本章算法误差绝对值');
xlabel('Sample number','FontSize',14);
ylabel('Absolute error / K','FontSize',14);
string = {'Class L'};
title(string);
set(gca,'FontSize',14);


xlim([1, 90]);
grid;