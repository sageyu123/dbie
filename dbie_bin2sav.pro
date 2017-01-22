pro dbie_bin2sav
; +
; NAME
;   DBIE_BIN2SAV
; PURPOSE:
;   Convert the result of extrapolation to IDL sav file
; SAMPLE CALLS:
;   IDL> dbie_make_boundary
; OUTPUTS:
; an IDL sav file containing the boundary magnetic field and the extrapolated 3-Dimension vector magteic field.
; HISTORY:
;   2014-12-12 - Sijie Yu (sjyu@nao.cas.cn or sijie.yu@njit.edu) 

path_results  = './results/'

filename = file_search(path_results+'mag.????????_??????_???x???_001t???_TAI.bin')
ndx = double(strmid(filename[0],strpos(filename[0],'x')-3,3))
ndy = double(strmid(filename[0],strpos(filename[0],'x')+1,3))
ndz = double(strmid(filename[0],strpos(filename[0],'_TAI')-3,3))+1

bx0 = fltarr(ndx,ndy,ndz)
by0 = fltarr(ndx,ndy,ndz)
bz0 = fltarr(ndx,ndy,ndz)

;filename = file_search(path_results+'mag.720s_CEA.20110215_010000_300x300_001t099_TAI.bin')  
nxnynz    = long(ndx)*long(ndy)*long(ndz-1)
date      = STRMID(filename, 38, 8, /REVERSE_OFFSET)
time      = STRMID(filename, 29, 6, /REVERSE_OFFSET)	

FOR nn  = 0, n_elements(filename)-1 DO BEGIN
	ind_h = where(date eq date[nn] and time eq time[nn])

	file_b = './boundary_data/boundary.'+date[nn]+'_'+time[nn]+'_TAI.bin'
  dummyfeld=fltarr(3L*ndx*ndy)
  Openr,1,file_b
  readu,1, dummyfeld
  close,1 
  dummyfeld = reform(dummyfeld,ndx,ndy,3)
	
  bx0[*,*,0] = dummyfeld[*,*,0]
  by0[*,*,0] = dummyfeld[*,*,1]
  bz0[*,*,0] = dummyfeld[*,*,2]
	
	FOR j        = 0,N_ELEMENTS(ind_h)-1 DO BEGIN		
  	height_start = long(STRMID(filename[ind_h[j]], 14, 3, /REVERSE_OFFSET))
  	height_end   = long(STRMID(filename[ind_h[j]], 10, 3, /REVERSE_OFFSET)) 
  	nz           = height_end-height_start+1
  	ind          = indgen(nz)*3
  	dummyfeld    = fltarr(long(ndx)*long(ndy)*nz*3)		
  	Openr,1,filename[ind_h[j]]
  	readu,1, dummyfeld
  	close,1	

  	dummyfeld = reform(dummyfeld,ndx,ndy,nz*3)
  	bx0[*,*,height_start:height_end] = dummyfeld[*,*,ind]
  	by0[*,*,height_start:height_end] = dummyfeld[*,*,ind+1]
  	bz0[*,*,height_start:height_end] = dummyfeld[*,*,ind+2]			
			
	ENDFOR
	print,filename[nn] + ' is readed'

	bx = bx0
	by = by0
	bz = bz0

  if ~file_test('./output/') $
  then file_mkdir,'./output/'
  savfile = './output/b3d.'+date[nn]+'_'+time[nn]+'.sav'
  save,filename=savfile,bx,by,bz,/VERBOSE
  print,'3D magnetic field saved to '+savfile
; stop
endfor

end
