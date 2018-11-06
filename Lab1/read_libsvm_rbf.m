function[vector,nsv,rho,gamma]=read_libsvm_rbf(str)

vector=0;

fp=fopen(str,'r');
rline=fscanf(fp,'%s ',1);
nsv=0;
gamma=0;
while strcmp(rline,'SV')~=1
    if strcmp(rline,'total_sv')==1
        rline=fscanf(fp,'%s ',1);
        nsv=str2num(rline);
    elseif strcmp(rline,'gamma')==1
        rline=fscanf(fp,'%s ',1);
        gamma=str2num(rline);
    elseif strcmp(rline,'rho')==1
        rline=fscanf(fp,'%s ',1);
        rho=str2num(rline);
    end;
    rline=fscanf(fp,'%s ',1);
end;

vector=zeros(nsv,3);

cnt=1;
while feof(fp)==0
    rline=fscanf(fp,'%s ',1);
    alpha=str2num(rline);
    rline0=fscanf(fp,'%s ',1);
    i0=findstr(rline0,':');
    x0=str2num(rline0(i0+1:end));
    rline1=fscanf(fp,'%s ',1);
    i1=findstr(rline1,':');
    x1=str2num(rline1(i1+1:end));
    
    vector(cnt,:)=[alpha x0 x1 ];
    cnt=cnt+1;
end;
  
fclose(fp);