%% png to raw
% open png file and write the result matrix to a 16 bit raw file
% note that matlab is column specific, so we inverted the matrix to row
% major
[I]=imread('ducks-16.png');
win=fopen('ducks-16.raw','w');
Z=I';
fwrite(win,Z,'*ubit16');

imshow(I);
%% draw raw files with different byte order
%read raw files by converting the byte order
clear
fin=fopen('lena16bit.raw');
I=fread(fin, [256 256],'*ubit16', 'b'); 
Z=I';
fclose(fin);

k=imshow(Z);
%% draw the output matrix in raw format file
%read the output matrix with added 2*3 elements to row and columns
%because the output binary file is in 32 bit convert it to 16 bit first

clear

% fin=fopen('lena.bin');
fin=fopen('ducks-16.bin');
% I=fread(fin, [262 262],'int32=>int16'); 
I=fread(fin, [2406 2406],'int32=>int16');
Z=I';
fclose(fin);

%multiply elements with a constant to get a grey picture 
% Z = Z*6000;

%black out the elements that are not zero and make 0 elements bigger to
%make them white
Z(Z==0)=50000;
Z(Z<9)=-50000;

k=imshow(Z);
