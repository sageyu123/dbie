pro dbie_make_boundary,bx,by,bz,date,time
; +
; NAME
;   DBIE_MAKE_BOUNDARY
; PURPOSE:
;   create an input binary file of boundary data.
; SAMPLE CALLS:
;   IDL> dbie_make_boundary,bx,by,bz
; OUTPUTS:
;	an binary file containing the vector magteic field data on the boundary.
; INPUTS:
;   REQUIRED:
;		bx: a 2D array of the x-component of the vector magnetic field on the boundary
;		by:	a 2D array of the y-component of the vector magnetic field on the boundary
;		bz:	a 2D array of the z-component of the vector magnetic field on the boundary
;       date: date of the observation in the format of yyyymmdd, e.g., '20120215'
;       time: time of the observation in the format of hhmmss, e.g., '120000'. 
;		(note: Please set the ss to 00 because the time cadence of HMI vector magnetic field products is 12 minutes.)
; HISTORY:
;   2014-12-12 - Sijie Yu (sjyu@nao.cas.cn or sijie.yu@njit.edu) 

if ~file_test('./boundary_data/') $
then file_mkdir,'./boundary_data/'

filename1='./boundary_data/boundary.'+date+'_'+time+'_TAI.bin'
print,'write ',filename1


nx = (size(bz))[1]
ny = (size(bz))[2]
nz = 1
nz=long(nz)
nxnynz=nx*ny*nz
nxny=nx*ny
dummyfeld=fltarr(3*nxnynz)
for ix=0, nx-1 do begin
	for iy=0, ny-1 do begin
		for iz=0, nz-1 do begin
		;iz=0
		i=ix+iy*nx+iz*nxny
		i2=i+nxnynz
		i3=i2+nxnynz
		dummyfeld[i]=bx[ix,iy,iz]
		dummyfeld[i2]=by[ix,iy,iz]
		dummyfeld[i3]=bz[ix,iy,iz]
		endfor
	endfor
endfor

Openw, 1,filename1
writeu,1, dummyfeld
close,1
; print,filename1+' saved  '
; print,bx[10,20,0]


END
