% The elastic wave equation in 2D
% Two subdomain (y>0 & y<0)
% Characteristic BC

% The semidiscreitzation is a first order system of ODEs
% We order them as u, u_t, v, w, q
% For each variable, there are two fields, i.e. u =[u1,u2]
close all
clear all

% Grid, upper domain 
n1 = 101; % Dof in y
m1 = round((n1-1)*1.1+1); % Dof in x, multiplying 1.1 to account for PML

N1  = m1*n1; 
Lmax = 4*pi; % Upper domain [0, Lmax]^2 (without PML)
delta1 = .1*Lmax; % PML width 
a1 = 0; b1 = Lmax+delta1; % x\in [a1, b1]
c1 = 0; d1 = Lmax; % x\in [c1, d1]
x1 = linspace(a1,b1,m1)'; % x-grid
y1 = linspace(c1,d1,n1)'; % y-grid
hx1 = x1(2)-x1(1); % grid size in x
hy1 = y1(2)-y1(1); % grid size in y
h1 = min(hx1,hy1); % grid size in domain 1
[X1,Y1] = meshgrid(x1,y1); % grid in domain 1
Xv1 = reshape(X1,N1,1); % vectorize grid in x
Yv1 = reshape(Y1,N1,1); % vectorize grid in y

% SBP operators in 1D, fully compatible
% [Hx1,Dx1,~,Sx1,Dxx1] = SBP4(m1,hx1);
% [Hy1,Dy1,~,Sy1,Dyy1] = SBP4(n1,hy1);
% Sx1(1,:)=Dx1(1,:);
% Sx1(end,:)=Dx1(end,:);
% Sy1(1,:)=Dy1(1,:);
% Sy1(end,:)=Dy1(end,:);

% SBP operators in 1D, non-fully compatible
[Hx1,Dx1,Dxx1,Sx1] = SBP4(m1,hx1);
[Hy1,Dy1,Dyy1,Sy1] = SBP4(n1,hy1);

% identity matrices
Ix1 = speye(m1);
Iy1 = speye(n1);
I2N1 = speye(2*N1);
IN1 = speye(N1);

% SBP operators in 2D
HxI1 = kron(inv(Hx1),Iy1);
Hx1 = kron(Hx1,Iy1);
Dx1 = kron(Dx1,Iy1);
Dxx1 = kron(Dxx1,Iy1);
Sx1 = kron(Sx1,Iy1);
HyI1 = kron(Ix1,inv(Hy1));
Hy1 = kron(Ix1,Hy1);
Dy1 = kron(Ix1,Dy1);
Dyy1 = kron(Ix1,Dyy1);
Sy1 = kron(Ix1,Sy1);
H1 = Hx1*Hy1;

% Trace operators
Exm1 = sparse(m1,m1); Exm1(m1,m1) = 1; Exm1 = kron(Exm1,Iy1);
Ex01 = sparse(m1,m1); Ex01(1,1) = 1; Ex01 = kron(Ex01,Iy1);

Eyn1 = sparse(n1,n1); Eyn1(n1,n1) = 1; Eyn1 = kron(Ix1,Eyn1);
Ey01 = sparse(n1,n1); Ey01(1,1) = 1; Ey01 = kron(Ix1,Ey01);

% Mixed derivative operators, 2D
Dxy1 = Dx1*Dy1;
Dyx1 = Dy1*Dx1;


% Material properties
% isotropic 
rho1 = 1.5;
mu1 = 1.8^2*rho1;
lambda1 = 3.118^2*rho1-2*mu1;
c11_1 = 2*mu1+lambda1;
c22_1 = c11_1;
c33_1 = mu1;
c12_1 = lambda1;

orthotropic = 1;
% orthotropic
if orthotropic == 1
    c11_1 = 4;
    c22_1 = 20;
    c33_1 = 2;
    c12_1 = 3.8;
end

A1 = [c11_1, 0
    0, c33_1];
B1 = [c33_1, 0
    0, c22_1];
C1 = [0, c12_1
    c33_1, 0];
Ct1 = C1'; 

% Second derivative with variable coefficients 
Qxx1 = kron(A1, Dxx1);
Qyy1 = kron(B1, Dyy1);
Qxy1 = kron(C1, Dxy1);
Qyx1 = kron(Ct1,Dyx1);


% Grid, lower domain 
m2 = m1; % Dof in x, has to be the same as m1
n2 = n1+0; % Dof in y 
N2  = m2*n2; 
delta2 = delta1; % PML width
a2 = 0; b2 = Lmax+delta2; % x\in [a2, b2]
c2 = -Lmax; d2 = 0; % x\in [c2, d2]
x2 = linspace(a2,b2,m2)'; % x grid
y2 = linspace(c2,d2,n2)'; % y grid 
hx2 = x2(2)-x2(1); % grid size in x
hy2 = y2(2)-y2(1); % grid size in y
h2 = min(hx2,hy2); % grid size in domain 2
h = min(h1,h2); % smallest grid size 
[X2,Y2] = meshgrid(x2,y2); % grid in domain 2
Xv2 = reshape(X2,N2,1); % vectorize grid in x
Yv2 = reshape(Y2,N2,1); % vectorize grid in y

% SBP operators in 1D, fully compatible
% [Hx2,Dx2,~,Sx2,Dxx2] = SBP4(m2,hx2);
% [Hy2,Dy2,~,Sy2,Dyy2] = SBP4(n2,hy2);
% Sx2(1,:)=Dx2(1,:);
% Sx2(end,:)=Dx2(end,:);
% Sy2(1,:)=Dy2(1,:);
% Sy2(end,:)=Dy2(end,:);

% SBP operators in 1D, non-fully compatible
[Hx2,Dx2,Dxx2,Sx2] = SBP4(m2,hx2);
[Hy2,Dy2,Dyy2,Sy2] = SBP4(n2,hy2);

% identity
Ix2 = speye(m2);
Iy2 = speye(n2);
I2 = speye(2);
I2N2 = speye(2*N2);
IN2  = speye(N2);
% SBP operators in 2D
HxI2 = kron(inv(Hx2),Iy2);
Hx2 = kron(Hx2,Iy2);
Dx2 = kron(Dx2,Iy2);
Dxx2 = kron(Dxx2,Iy2);
Sx2 = kron(Sx2,Iy2);
HyI2 = kron(Ix2,inv(Hy2));
Hy2 = kron(Ix2,Hy2);
Dy2 = kron(Ix2,Dy2);
Dyy2 = kron(Ix2,Dyy2);
Sy2 = kron(Ix2,Sy2);
H2 = Hx2*Hy2;

% Trace operators
Exm2 = sparse(m2,m2); Exm2(m2,m2) = 1; Exm2 = kron(Exm2,Iy2);
Ex02 = sparse(m2,m2); Ex02(1,1) = 1; Ex02 = kron(Ex02,Iy2);

Eyn2 = sparse(n2,n2); Eyn2(n2,n2) = 1; Eyn2 = kron(Ix2,Eyn2);
Ey02 = sparse(n2,n2); Ey02(1,1) = 1; Ey02 = kron(Ix2,Ey02);

EyInt12 = sparse(n1,n2); EyInt12(1,n2) = 1; EyInt12 = kron(Ix2,EyInt12); 

EyInt21 = sparse(n2,n1); EyInt21(n2,1) = 1; EyInt21 = kron(Ix1,EyInt21); 

Dxy2 = Dx2*Dy2;
Dyx2 = Dy2*Dx2;

% Material properties
% isotropic 
rho2 = 3;
mu2 = 3^2*rho2;
lambda2 = 5.196^2*rho2-2*mu2;
c11_2 = 2*mu2+lambda2;
c22_2 = c11_2;
c33_2 = mu2;
c12_2 = lambda2;

% orthotropic = 0;

mps = 4; % ratio between material properties in two domains 
if orthotropic == 1
    c11_2 = 4*mps;
    c22_2 = 20*mps;
    c33_2 = 2*mps;
    c12_2 = 3.8*mps;
end


A2 = [c11_2, 0
    0, c33_2];
B2 = [c33_2, 0
    0, c22_2];
C2 = [0, c12_2
    c33_2, 0];
Ct2 = C2'; 

Qxx2 = kron(A2, Dxx2);
Qyy2 = kron(B2, Dyy2);
Qxy2 = kron(C2, Dxy2);
Qyx2 = kron(Ct2,Dyx2);


% PML coefficient, domain 1
Ref1 = 1e-4; % magnitude of the relative modeling error 
cp1 = sqrt(c22_1/rho1);
cs1 = sqrt(c33_1/rho1);
cc1 = sqrt(cp1^2+cs1^2);
d0 = 4*cp1*log(1/Ref1)/(2*delta1); % d0 is damping strength

alpha1 = 0.05*d0;
disp(['Shear wave speed in doamin 1 is ' num2str(cs1)])
disp(['Pressure wave speed in doamin 1 is ' num2str(cp1)])

% PML coefficient, domain 2
Ref2 = 1e-4; % magnitude of the relative modeling error 
cp2 = sqrt(c22_2/rho2);
cs2 = sqrt(c33_2/rho2);
cc2 = sqrt(cp2^2+cs2^2);
disp(['Shear wave speed in doamin 2 is ' num2str(cs2)])
disp(['Pressure wave speed in doamin 2 is ' num2str(cp2)])

d0 = 4*cp2*log(1/Ref2)/(2*delta2); % d0 is damping strength

alpha2 = 0.05*d0;
% Make sure we use one alpha in the entire domain
if alpha2<alpha1
    alpha2 = alpha1;
else
    alpha1 = alpha2;
end

% Remove dampling
disp(' ')
rd = 0;
if rd == 1
  d0 = 0; alpha1=0; alpha2=0;
  disp('PML dampling removed')
else
  disp('PML dampling included')  
end
disp(' ')

SigmaV1 = d0*((Xv1-Lmax)/delta1).^3*1+0*d0;
SigmaV1(SigmaV1<0)=0;
Sigma1 = sparse(diag(SigmaV1));
figure,surf(X1,Y1,reshape(SigmaV1,n1,m1))

SigmaV2 = d0*((Xv2-Lmax)/delta2).^3*1+0*d0;
SigmaV2(SigmaV2<0)=0;
Sigma2 = sparse(diag(SigmaV2));
hold on,surf(X2,Y2,reshape(SigmaV2,n2,m2))
xlabel('X')

% Discretization matrix, [u,ut,v,w,q]

Q1 = [sparse(2*N1,2*N1), speye(2*N1,2*N1),sparse(2*N1,6*N1) % u
    Qxx1+Qyy1+Qxy1+Qyx1+kron(I2,Sigma1)*alpha1*rho1, -kron(I2,Sigma1)*rho1, -kron(A1, Dx1)*kron(I2,Sigma1),kron(B1, Dy1)*kron(I2,Sigma1), -kron(I2,Sigma1)*alpha1*rho1
    kron(I2, Dx1), sparse(2*N1,2*N1), -kron(I2,Sigma1)-alpha1*I2N1,sparse(2*N1,4*N1)
    kron(I2, Dy1), sparse(2*N1,4*N1), -alpha1*I2N1, sparse(2*N1,2*N1)
    alpha1*I2N1, sparse(2*N1,6*N1),-alpha1*I2N1];

Q2 = [sparse(2*N2,2*N2), speye(2*N2,2*N2),sparse(2*N2,6*N2) % u
    Qxx2+Qyy2+Qxy2+Qyx2+kron(I2,Sigma2)*alpha2*rho2, -kron(I2,Sigma2)*rho2, -kron(A2, Dx2)*kron(I2,Sigma2),kron(B2, Dy2)*kron(I2,Sigma2), -kron(I2,Sigma2)*alpha2*rho2
    kron(I2, Dx2), sparse(2*N2,2*N2), -kron(I2,Sigma2)-alpha1*I2N2,sparse(2*N2,4*N2)
    kron(I2, Dy2), sparse(2*N2,4*N2), -alpha1*I2N2, sparse(2*N2,2*N2)
    alpha2*I2N2, sparse(2*N2,6*N2),-alpha2*I2N2];


% SAT, domain 1
% BC in the upper domain
% top: characteristic
% left and right: characteristic
Z1x = [speye(N1,N1)*cp1*rho1, sparse(N1,N1)
sparse(N1,N1), speye(N1,N1)*cs1*rho1];
Z1y = [speye(N1,N1)*cs1*rho1, sparse(N1,N1)
 sparse(N1,N1), speye(N1,N1)*cp1*rho1];

tau1 = -1; tau2 = -1; tau3 = -1; tau4 = -1;
r = 1;
SAT1 = sparse(10*N1,10*N1);
P1 = tau1*kron(I2,HxI1*Exm1)*(kron(A1,Sx1)+kron(C1,Dy1)) ...
    -tau2*kron(I2,HxI1*Ex01)*(kron(A1,Sx1)+kron(C1,Dy1)) ...
    +tau3*kron(I2,HyI1*Eyn1)*(kron(B1,Sy1)+kron(Ct1,Dx1)) ...
    -0*tau4*kron(I2,HyI1*Ey01)*(kron(B1,Sy1)+kron(Ct1,Dx1));
P2 = Z1x*(tau1*kron(I2,HxI1*Exm1)*r + tau2*kron(I2,HxI1*Ex01)*r) ...
    + Z1y*tau3*kron(I2,HyI1*Eyn1)*r*1 + 0*tau4*kron(I2,HyI1*Ey01)*r;

% P3 corresponds to the modification in free surface in PML, i.e. variable w
P3 = tau3*kron(I2,HyI1*Eyn1)*kron(B1,IN1)*kron(I2,Sigma1);  
% P4 corresponds to the modification in characteristic BC (left & right) in PML, i.e. variable v
P4 = -tau1*kron(I2,HxI1*Exm1)*kron(A1,Sigma1)+tau2*kron(I2,HxI1*Ex01)*kron(A1,Sigma1);

% u-q part
P1 = P1+tau3*kron(I2,Sigma1)*Z1y*kron(I2,HyI1*Eyn1);
P5 = -tau3*kron(I2,Sigma1)*Z1y*kron(I2,HyI1*Eyn1);

% modi_PML_fs = 1;
% if modi_PML_fs ~= 1
%     P3=P3*0;
%     disp('No modification of free surface BC in PML, domain 1')
% else
%     disp('Use modified free surface SAT in PML, domain 1')
% end

SAT1(2*N1+1:4*N1, 1:2*N1) = P1;
SAT1(2*N1+1:4*N1, 2*N1+1:4*N1) = P2;
SAT1(2*N1+1:4*N1, 6*N1+1:8*N1) = P3;
SAT1(2*N1+1:4*N1, 4*N1+1:6*N1) = P4;
SAT1(2*N1+1:4*N1, 8*N1+1:10*N1) = P5;


% return
% interface SAT, domain 1
% penalty parameter
tau = 20*(cc1+cc2);

SATint1 = sparse(10*N1,10*N1);
KV = 0;
if KV == 1
    SATint1(2*N1+1:4*N1,1:2*N1) = ...
        1/2*kron(I2,HyI1*Ey01)*(kron(B1,Sy1)+kron(Ct1,Dx1)) ...
        -1/2*kron(I2,HyI1)*kron(I2,HxI1)*(kron(B1,Sy1)+kron(Ct1,Dx1))'*kron(I2,Hx1)*kron(I2,Ey01) ...
        -tau/h*kron(I2,HyI1)*kron(I2,Ey01);
    disp('Use Kristoffers trick, multiplying H^-1 and then H, domian 1')
else
    SATint1(2*N1+1:4*N1,1:2*N1) = ...
        1/2*kron(I2,HyI1*Ey01)*(kron(B1,Sy1)+kron(Ct1,Dx1)) ...
        -1/2*kron(I2,HyI1)*(kron(B1,Sy1)+kron(Ct1,Dx1))'*kron(I2,Ey01) ...
        -tau/h*kron(I2,HyI1)*kron(I2,Ey01);
    disp('No Kristoffers trick, multiplying H^-1 and then H, domian 1')
end
SATint12 = sparse(10*N1,10*N2);
if KV == 1
    SATint12(2*N1+1:4*N1,1:2*N2) = ...
        -1/2*kron(I2,HyI1*EyInt12)*(kron(B2,Sy2)+kron(Ct2,Dx2)) ...
        +1/2*kron(I2,HyI1)*kron(I2,HxI1)*(kron(B1,Sy1)+kron(Ct1,Dx1))'*kron(I2,Hx1)*kron(I2,EyInt12) ...
        +tau/h*kron(I2,HyI1)*kron(I2,EyInt12);
else
    SATint12(2*N1+1:4*N1,1:2*N2) = ...
        -1/2*kron(I2,HyI1*EyInt12)*(kron(B2,Sy2)+kron(Ct2,Dx2)) ...
        +1/2*kron(I2,HyI1)*(kron(B1,Sy1)+kron(Ct1,Dx1))'*kron(I2,EyInt12) ...
        +tau/h*kron(I2,HyI1)*kron(I2,EyInt12);
end

% modify traction for interface, upper domain
modi_PML_int = 1;
if modi_PML_int ~= 1
    SATint1(2*N1+1:4*N1,6*N1+1:8*N1) = 0/2*kron(I2,HyI1*Ey01)*kron(B1,IN1)*kron(I2,Sigma1);
    SATint12(2*N1+1:4*N1,6*N2+1:8*N2) = ...
        -0/2*kron(I2,HyI1*EyInt12)*(kron(B2,IN2)*kron(I2,Sigma2));
    disp('No modification of interface SAT in PML, domian 1')
else
    SATint1(2*N1+1:4*N1,6*N1+1:8*N1) = 1/2*kron(I2,HyI1*Ey01)*kron(B1,IN1)*kron(I2,Sigma1);
    SATint12(2*N1+1:4*N1,6*N2+1:8*N2) = ...
        -1/2*kron(I2,HyI1*EyInt12)*(kron(B2,IN2)*kron(I2,Sigma2));
    disp('Use modified interface SAT in PML, domian 1')
end
disp(' ')

% SAT for BC, domain 2
% bottom, left and right: characteristic BC
Z2x = [speye(N2,N2)*cp2*rho2, sparse(N2,N2)
sparse(N2,N2), speye(N2,N2)*cs2*rho2];
Z2y = [speye(N2,N2)*cs2*rho2, sparse(N2,N2)
sparse(N2,N2), speye(N2,N2)*cp2*rho2];
tau1 = -1; tau2 = -1; tau3 = -1; tau4 = -1;
r = 1;
SAT2 = sparse(10*N2,10*N2);
P1 = tau1*kron(I2,HxI2*Exm2)*(kron(A2,Sx2)+kron(C2,Dy2)) ...
    -tau2*kron(I2,HxI2*Ex02)*(kron(A2,Sx2)+kron(C2,Dy2)) ...
    +0*tau3*kron(I2,HyI2*Eyn2)*(kron(B2,Sy2)+kron(Ct2,Dx2)) ...
    -tau4*kron(I2,HyI2*Ey02)*(kron(B2,Sy2)+kron(Ct2,Dx2));
P2 = Z2x*(tau1*kron(I2,HxI2*Exm2)*r + tau2*kron(I2,HxI2*Ex02)*r) ...
    + Z2y*(0*tau3*kron(I2,HyI2*Eyn2)*r + tau4*kron(I2,HyI2*Ey02)*r);
% P3 corresponds to the modification in characteristic BC (bottom) in PML, i.e. variable w
P3 = -tau4*kron(I2,HyI2*Ey02)*kron(B2,IN2)*kron(I2,Sigma2); 
% P4 corresponds to the modification in characteristic BC (left & right) in PML, i.e. variable v
P4 = -tau1*kron(I2,HxI2*Exm2)*kron(A2,Sigma2)+tau2*kron(I2,HxI2*Ex02)*kron(A2,Sigma2);


% u-q part
P1 = P1+tau4*kron(I2,Sigma2)*Z2y*kron(I2,HyI2*Ey02);
P5 = -tau4*kron(I2,Sigma2)*Z2y*kron(I2,HyI2*Ey02);




% modi_PML_fs = 1;
% if modi_PML_fs ~= 1
%     P3=P3*0;
%     disp('No modification of characteristic BC in PML, domain 2')
% else
%     disp('Use modified characteristic SAT in PML, domain 2')
% end

SAT2(2*N2+1:4*N2, 1:2*N2) = P1;
SAT2(2*N2+1:4*N2, 2*N2+1:4*N2) = P2;
SAT2(2*N2+1:4*N2, 6*N2+1:8*N2) = P3;
SAT2(2*N2+1:4*N2, 4*N2+1:6*N2) = P4;

SAT2(2*N2+1:4*N2, 8*N2+1:10*N2) = P5;


% interface SAT
SATint2 = sparse(10*N2,10*N2);
if KV == 1
 SATint2(2*N2+1:4*N2,1:2*N2) = ...
     -1/2*kron(I2,HyI2*Eyn2)*(kron(B2,Sy2)+kron(Ct2,Dx2)) ...
     +1/2*kron(I2,HyI2)*kron(I2,HxI2)*(kron(B2,Sy2)+kron(Ct2,Dx2))'*kron(I2,Hx2)*kron(I2,Eyn2) ...
     -tau/h*kron(I2,HyI2)*kron(I2,Eyn2);
    disp('Use Kristoffers trick, multiplying H^-1 and then H, in domain 2')
else
    SATint2(2*N2+1:4*N2,1:2*N2) = ...
    -1/2*kron(I2,HyI2*Eyn2)*(kron(B2,Sy2)+kron(Ct2,Dx2)) ...
    +1/2*kron(I2,HyI2)*(kron(B2,Sy2)+kron(Ct2,Dx2))'*kron(I2,Eyn2) ...
    -tau/h*kron(I2,HyI2)*kron(I2,Eyn2);
    disp('No Kristoffers trick, multiplying H^-1 and then H, in domain 2')
end
SATint21 = sparse(10*N2,10*N1);
if KV == 1
 SATint21(2*N2+1:4*N2,1:2*N1) = ...
     +1/2*kron(I2,HyI2*EyInt21)*(kron(B1,Sy1)+kron(Ct1,Dx1)) ...
     -1/2*kron(I2,HyI2)*kron(I2,HxI2)*(kron(B2,Sy2)+kron(Ct2,Dx2))'*kron(I2,Hx2)*kron(I2,EyInt21) ...
     +tau/h*kron(I2,HyI2)*kron(I2,EyInt21);
else
    SATint21(2*N2+1:4*N2,1:2*N1) = ...
    +1/2*kron(I2,HyI2*EyInt21)*(kron(B1,Sy1)+kron(Ct1,Dx1)) ...
    -1/2*kron(I2,HyI2)*(kron(B2,Sy2)+kron(Ct2,Dx2))'*kron(I2,EyInt21) ...
    +tau/h*kron(I2,HyI2)*kron(I2,EyInt21);
end

% modify traction for interface, lower domain
if modi_PML_int ~= 1
    SATint2(2*N2+1:4*N2,6*N2+1:8*N2) = ...
        -0/2*kron(I2,HyI2*Eyn2)*(kron(B2,IN2)*kron(I2,Sigma2));
    SATint21(2*N2+1:4*N2,6*N1+1:8*N1) = ...
        +0/2*kron(I2,HyI2*EyInt21)*(kron(B1,IN1)*kron(I2,Sigma1)); 
    disp('No modification of interface SAT in PML, domian 2')
else
    SATint2(2*N2+1:4*N2,6*N2+1:8*N2) = ...
        -1/2*kron(I2,HyI2*Eyn2)*(kron(B2,IN2)*kron(I2,Sigma2));
    SATint21(2*N2+1:4*N2,6*N1+1:8*N1) = ...
        +1/2*kron(I2,HyI2*EyInt21)*(kron(B1,IN1)*kron(I2,Sigma1)); 
    disp('Use modifified interface SAT in PML, domian 2')
end

QSAT1 = Q1+SAT1+SATint1*1;
QSAT2 = Q2+SAT2+SATint2*1;

% Take care of the densities
RHO1INV = speye(2*N1,2*N1)*1/rho1;
RHO2INV = speye(2*N2,2*N2)*1/rho2;

% QSAT is SBP+SAT for the entire domain
QSAT = sparse(10*N1+10*N2,10*N1+10*N2);

QSAT(1:10*N1, 1:10*N1) = QSAT1;
QSAT(10*N1+1:end,10*N1+1:end) = QSAT2;

QSAT(1:10*N1, 10*N1+1:end) = SATint12*1;
QSAT(10*N1+1:end,1:10*N1) = SATint21*1;


QSAT(10*N1+2*N2+1:10*N1+4*N2, 1:end) = RHO2INV*QSAT(10*N1+2*N2+1:10*N1+4*N2, 1:end);
QSAT(2*N1+1:4*N1, 1:end) = RHO1INV*QSAT(2*N1+1:4*N1, 1:end);
 

% % remove the auxiliary variables in the eig value analysis for the model without PML
% QSAT(4*N1+1:10*N1,:)=0;
% QSAT(10*N1+4*N2+1:end,:)=0;
% 
% disp('Eigenvalues...')
% disp(['Size of QSAT is ' num2str(size(QSAT))])
% eigD = eig(full(QSAT));
% sorted_eigD = sort(eigD,'descend', 'ComparisonMethod','real');
% figure,plot(real(eigD), imag(eigD), '*')
% sorted_eigD(1:5) 
% % eigs(QSAT,6,'bothendsreal','Tolerance',1e-5,'SubspaceDimension',100)
% figure,spy(QSAT)
% return
%  


% RK4
% Inject Dirichlet at the PML boundary, not used anymore
% x_upper_index = [N1-n1+1:N1, 2*N1-n1+1:2*N1, 3*N1-n1+1:3*N1, 4*N1-n1+1:4*N1, 5*N1-n1+1:5*N1, ...
% 6*N1-n1+6:N1, 7*N1-n1+1:7*N1, 8*N1-n1+1:8*N1, 9*N1-n1+1:9*N1, 10*N1-n1+1:10*N1];
% 
% x_lower_index = 10*N1+[N2-n2+1:N2, 2*N2-n2+1:2*N2, 3*N2-n2+1:3*N2, 4*N2-n2+1:4*N2, 5*N2-n2+1:5*N2, ...
% 6*N2-n2+6:N2, 7*N2-n2+1:7*N2, 8*N2-n2+1:8*N2, 9*N2-n2+1:9*N2, 10*N2-n2+1:10*N2];
% 
% x1_index = [x_upper_index, x_lower_index];

% % Initial condition
u11 = exp(20*(-(Xv1-Lmax/2).^2-(Yv1-2*Lmax/5).^2));
u21 = exp(20*(-(Xv2-Lmax/2).^2-(Yv2-2*Lmax/5).^2));
u12 = u11;
u22 = u21;

% parameters, Stoneley
pB2 = -3.370750095768126; pA1 = 1.792135164314052; pB1 = -6.044875895813948;
c=0.315111729874378;


r = sqrt(1-c^2*rho1/(lambda1+2*mu1));
rp = sqrt(1-c^2*rho2/(lambda2+2*mu2));
s = sqrt(1-c^2*rho1/mu1);
sp = sqrt(1-c^2*rho2/mu2);


% Domain 1
% t=0;
% u11 = (exp(-r*Yv1)-s*pB2*exp(-s*Yv1)).*cos(Xv1-c*t).*exp(-0*(Xv1-2*pi).^2);
% u12 = (-r*exp(-r*Yv1)+pB2*exp(-s*Yv1)).*sin(Xv1-c*t).*exp(-0*(Xv1-2*pi).^2);
% % Domain 2
% u21 = (pA1*exp(rp*Yv2)+sp*pB1*exp(sp*Yv2)).*cos(Xv2-c*t).*exp(-0*(Xv2-2*pi).^2);
% u22 = (rp*pA1*exp(rp*Yv2)+pB1*exp(sp*Yv2)).*sin(Xv2-c*t).*exp(-0*(Xv2-2*pi).^2);


% u11 = Xv1*0+1;
% u12 = u11;
% 
% u21 = Xv2*0+1;
% u22 = u21;

%  u11(Xv1>=Lmax/1) = 0;
%  u12(Xv1>=Lmax/1) = 0;
%  u21(Xv2>=Lmax/1) = 0;
%  u22(Xv2>=Lmax/1) = 0;

% z contains the solution + auxiliary variables
z = [u11;u12;zeros(8*N1,1);u21;u22;zeros(8*N2,1)];


dt = 0.2*h/max(cc1,cc2); % stepsize in time
T = 10; % final time 
nt = ceil(T/dt); % no. of time steps
dt = T/nt; % stepsize in time
energy = zeros(nt,1);

% plot solutions at the following time points
tsave = [0:0.2:10, 11:300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000];
tsave_ind = 1;


% forcing 
F = zeros(10*(N1+N2),1);

pause(0.1)
figure

% start time loop
for i = 1:nt
    t = (i-1)*dt;
    
    G = zeros(10*(N1+N2),1);
    K1 = QSAT*z+G;
    
    t = (i-1/2)*dt;
    
    K2 = QSAT*(z+dt/2*K1)+G;
    K3 = QSAT*(z+dt/2*K2)+G;
    
    t = i*dt;
    K4 = QSAT*(z+dt*K3)+G;
    
    z = z + dt/6*(K1+2*K2+2*K3+K4);
%     z(x1_index) = 0; % Inject zero Dirichlet at PML boundary
    u1 = sqrt(0.5*(reshape(z(1:N1),n1,m1).^2 + reshape(z(N1+1:2*N1),n1,m1).^2));
    u2 = sqrt(0.5*(reshape(z(10*N1+1:10*N1+N2),n2,m2).^2 + reshape(z(10*N1+N2+1:10*N1+2*N2),n2,m2).^2));
    
    
    
    if t+1e-10>tsave(tsave_ind)
%         figure
        surf(X1,Y1,u1);
        hold on
        surf(X2,Y2,u2);
        hold off
        axis equal
        shading interp
        %     caxis([-.2,.2]);
        colorbar
        colormap jet
        view(2)
        %title(['time is ' num2str(t) ', max is ' num2str(max(abs(z(1:2*N1))))])
%         title(['$t$=' num2str(round(t))],'Interpreter','latex','fontsize',16)
        ax = gca;
        ax.FontSize = 16; 
        drawnow

%         saveas(gcf,['TwoDomainIso' '_t_' num2str(tsave(tsave_ind)) '.png'])
        ufinal_PML(:,tsave_ind) = [z(1:n1^2); z(N1+1:N1+n1^2); z(10*N1+1:10*N1+n2^2); z(10*N1+N2+1:10*N1+N2+n2^2)];

        tsave_ind = tsave_ind+1;

    end

    energy(i) = sqrt(z(1:N1)'*H1*z(1:N1)+z(N1+1:2*N1)'*H1*z(N1+1:2*N1) ...
        +z(10*N1+1:10*N1+N2)'*H2*z(10*N1+1:10*N1+N2)+z(10*N1+N2+1:10*N1+2*N2)'*H2*z(10*N1+N2+1:10*N1+2*N2));

end

%ufinal_PML = [z(1:n1^2); z(N1+1:N1+n1^2); z(10*N1+1:10*N1+n2^2); z(10*N1+N2+1:10*N1+N2+n2^2)];

% figure, plot(linspace(0,T,nt), energy)

% figure, semilogy(linspace(0,T,nt), energy,'linewidth',3)
% xlabel('$t$','Interpreter','latex','fontsize',16)
% ylabel('$\|\mathbf{u}\|_H$','Interpreter','latex','fontsize',16)
% ax = gca;
% ax.FontSize = 16;
% 
% saveas(gcf,[ 'TwoDomainsIso_m' num2str(m1) '_energy.png'])
% t_vec=linspace(0,T,nt);
% save('TwoDomainIsoEnergyData','t_vec','energy')

