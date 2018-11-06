function[]=write_svm_file(mat1,mat0,str)

sz=size(mat1);

fp=fopen(str,'w');

for i=1:sz(1)
	fprintf(fp,'+1 ');
	for j=1:sz(2)
		fprintf(fp,'%d:%f ',j,mat1(i,j));
    end
	fprintf(fp,' # data %d \n',i);
end

sz=size(mat0);

for i=1:sz(1)
	fprintf(fp,'-1 ');
    for j=1:sz(2)
		fprintf(fp,'%d:%f ',j,mat0(i,j));
    end
	fprintf(fp,' # data %d \n',i);
end

fclose(fp);


