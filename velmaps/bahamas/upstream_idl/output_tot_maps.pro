function getPositions, numpart_total, pos_fof, $
                       pos_dm, pos_gas, pos_star, $
                       mdm, mgas, mstar, $
                       ifof, h, projection, x=x, y=y, z=z

  if numpart_total[0] gt 0l and numpart_total[4] gt 0l then begin
     xFrame=[reform(pos_dm[0,*])-pos_fof[0,ifof], $
        reform(pos_gas[0,*])-pos_fof[0,ifof],$
        reform(pos_star[0,*])-pos_fof[0,ifof]]

     yFrame=[reform(pos_dm[1,*])-pos_fof[1,ifof],$
        reform(pos_gas[1,*])-pos_fof[1,ifof],$
        reform(pos_star[1,*])-pos_fof[1,ifof]]
     
     zFrame=[reform(pos_dm[2,*])-pos_fof[2,ifof],$
        reform(pos_gas[2,*])-pos_fof[2,ifof],$
        reform(pos_star[2,*])-pos_fof[2,ifof]]
     mtmp=dblarr(n_elements(reform(pos_dm[0,*])))
     mtmp[*]=mdm
        
     m=[mtmp,mgas,mstar]
        
     m*=1d10/h
  endif
  if numpart_total[0] gt 0l  and numpart_total[4] eq 0l then begin
     xFrame=[reform(pos_dm[0,*])-pos_fof[0,ifof],$
        reform(pos_gas[0,*])-pos_fof[0,ifof]]

     yFrame=[reform(pos_dm[1,*])-pos_fof[1,ifof],$
        reform(pos_gas[1,*])-pos_fof[1,ifof]]
     zFrame=[reform(pos_dm[2,*])-pos_fof[2,ifof],$
        reform(pos_gas[2,*])-pos_fof[2,ifof]]
        
     mtmp=dblarr(n_elements(reform(pos_dm[0,*])))
     mtmp[*]=mdm

     m=[mtmp,mgas]
        
     m*=1d10/h
  endif
  if numpart_total[0] eq 0l  and numpart_total[4] eq 0l then begin
     xFrame=reform(pos_dm[0,*])-pos_fof[0,ifof]
     yFrame=reform(pos_dm[1,*])-pos_fof[1,ifof]
     zFrame=reform(pos_dm[2,*])-pos_fof[2,ifof]
     m=dblarr(n_elements(reform(pos_dm[0,*])))
     m[*]=mdm*1d10/h
  endif
  if projection eq 1 then begin
     x=xFrame
     y=yFrame
     z=zFrame
  endif
  if projection eq 2 then begin
     x=yFrame
     y=zFrame
     z=xFrame
  endif 
  if projection eq 3 then begin
     x=zFrame
     y=xFrame
     z=yFrame
  endif
  return,m
end

pro output_tot_maps, sims=sims, $
                     res=res, $
                     redshift=redshift, $
                     LOMASS=LOMASS, projection=projection
if not keyword_set(projection) then projection=3
if not keyword_set(redshift) then redshift=0.25
if not keyword_set(sims) then sims=['SIDM1+baryons']
if not keyword_set(res) then res='lo'
if not keyword_set(LOMASS) then LOMASS=''


msun=1.989d33
mpc=3.0857d24

dpix=0.001d0   ; pixel length in physical Mpc
fov=2d0   ; image FOV in physical Mpc
dz=10d0    ; depth of z-axis (for projection) in physical Mpc
npix=long(fov/dpix)  ; number of pixels

print,npix

;window,xsize=768,ysize=768
;loadct,3




IF SIMS EQ 'CDM_hi' THEN begin
   BASEDIR='/cosma7old/ICC-data/mccarthy/FOR_DAVID/BAHAMAS_hi_AGN_L400N1024_WMAP9/data/'
endif else $
   if SIMS EQ 'CDM_low' then begin
   BASEDIR='/cosma7old/ICC-data/mccarthy/FOR_DAVID/BAHAMAS_low_AGN_L400N1024_WMAP9/data/'
endif else $
   if SIMS EQ 'CDM' then begin
   BASEDIR='/cosma7old/ICC-data/mccarthy/cosma5/BAHAMAS/L400N1024/WMAP9/nu0/DMONLY/data/'
endif else $
   BASEDIR='/cosma7old/data/dp004/ndcf31/BAHAMAS/sims/L400N1024_WMAP9'



;redshift =0.1 = 0 but cant have a keyword as 0                                 
if redshift eq 0.1 then SNAP=32
if redshift eq 0.125 then snap=31
if redshift eq 0.250 then snap=30
if redshift eq 0.375 then snap=29
if redshift eq 0.5 then snap=28
if redshift eq 0.75 then snap=27
if redshift eq 1.0 then snap=26

nsim=n_elements(sims)

for isim=0,nsim-1 do begin

  simname=sims[isim]

  for isnap=snap, snap,-1 do begin


     if Sims eq 'CDM_low' or Sims eq 'CDM_hi' or Sims eq 'CDM' then $
        SIMDATADIR=BASEDIR $
     else $
        SIMDATADIR=BASEDIR+'/'+SIMNAME+'/data/'
     baseOutputDir = '/cosma7/data/dp004/dc-harv1/bahamas'
     IF ISNAP EQ 32 THEN OUTPUTDIR=baseOutputDir+'/'+simname+LOMASS+'/z_0.000/'
     if isnap eq 31 then outputdir=baseOutputDir+'/'+simname+LOMASS+'/z_0.125/'
     if isnap eq 30 then outputdir=baseOutputDir+'/'+simname+LOMASS+'/z_0.250/'
     if isnap eq 29 then outputdir=baseOutputDir+'/'+simname+LOMASS+'/z_0.375/'
     if isnap eq 28 then outputdir=baseOutputDir+'/'+simname+LOMASS+'/z_0.500/'
     if isnap eq 27 then outputdir=baseOutputDir+'/'+simname+LOMASS+'/z_0.750/'
     if isnap eq 26 then outputdir=baseOutputDir+'/'+simname+LOMASS+'/z_1.000/'

  spawn,'mkdir '+simname
  spawn,'mkdir '+outputdir

  print,simname
  infile_subtab=simdatadir+'groups_0'+strtrim(string(isnap),1)+'/eagle_subfind_tab_0'+strtrim(string(isnap),1)+'.0.hdf5'
  infile=simdatadir+'particledata_0'+strtrim(string(isnap),1)+'/eagle_subfind_particles_0'+strtrim(string(isnap),1)+'.0.hdf5'
  print,infile

  zred=h5ra(infile,'Header','Redshift')
  h=h5ra(infile,'Header','HubbleParam')
  Omega0=h5ra(infile,'Header','Omega0')
  OmegaLambda=h5ra(infile,'Header','OmegaLambda')
  aexp=h5ra(infile,'Header','ExpansionFactor')
  boxsize=h5ra(infile,'Header','BoxSize')
  OmegaBaryon=h5ra(infile,'Header','OmegaBaryon')
  masstable=h5ra(infile,'Header','MassTable')
  NumPart_Total=h5ra(infile,'Header','NumPart_Total')


;  rhocrit_z0=(1.88d-29*h*h)*(3.0857d24)^3d0/1.989d33  ; in Msun/Mpc3
;  rhocrit_z=rhocrit_z0*(OmegaLambda+Omega0*(1d0+zred)^3d0)

  print,zred,fov

  mdm=float(reform(masstable[1]))
  print,'   Reading subfind catalogs...'
;  M500 = READ_SUBFIND_EAGLE(infile_subtab,'FOF/Group_M_Crit500',/silent)/msun
;  R500 = READ_SUBFIND_EAGLE(infile_subtab,'FOF/Group_R_Crit500',/silent)/mpc
  M200 = READ_SUBFIND_EAGLE(infile_subtab,'FOF/Group_M_Crit200',/silent)/msun

  R200 = READ_SUBFIND_EAGLE(infile_subtab,'FOF/Group_R_Crit200',/silent)/mpc
  print,'   done.'

  nfof=n_elements(M200)

  pos_sub=READ_SUBFIND_EAGLE(infile_subtab,'Subhalo/CentreOfPotential',/silent,/NOCONVERSION)
  FirstSubhaloID=long(READ_SUBFIND_EAGLE(infile_subtab,'FOF/FirstSubhaloID',/SILENT))
  pos_fof=pos_sub[*,FirstSubhaloID]
  undefine,pos_sub
  undefine,FirstSubhaloID


  spawn,'ls '+outputdir+'/GALAXY_CATALOGS/cluster_*_galaxy_catalog.dat', clusters
  fof_groups = []
  for iCl=0, n_elements(clusters)-1 do begin
     readcol,clusters[iCl],fof,x,y,Ms,Mss,Msss,flag, /silent

     fof_groups = [ fof_groups, fof[0] ]
  endfor

  
  ncl = n_elements(fof_groups)
  nfof = n_elements(ncl)

  print,ncl,nfof

  if (numpart_total[4] gt 0l) then begin
    ispecies=4
    var='Mass'
    mstar=nread_gadget_hdf5(infile,ispecies,var,INPUT='snapshot',/silent,/NOCONVERSION)

    var='Coordinates'
    pos_star=nread_gadget_hdf5(infile,ispecies,var,INPUT='snapshot',/silent,/NOCONVERSION)
  endif
  if  (numpart_total[0] gt 0l) then begin
    ispecies=0
    var='Mass'
    mgas=nread_gadget_hdf5(infile,ispecies,var,INPUT='snapshot',/silent,/NOCONVERSION)

    var='Coordinates'
    pos_gas=nread_gadget_hdf5(infile,ispecies,var,INPUT='snapshot',/silent,/NOCONVERSION)
  endif

  ispecies=1
  var='Coordinates'
  pos_dm=nread_gadget_hdf5(infile,ispecies,var,INPUT='snapshot',/silent,/NOCONVERSION)
  print,'done reading snap data'

  c=0



  spawn,'mkdir '+outputdir+'/HIRES_MAPS'


  for icl=0l,ncl-1l do begin
     ifof = fix(fof_groups[icl])
     c=ifof

     outfile=outputdir+'/HIRES_MAPS/cluster_'+strtrim(c,2)+'_total_sph.fits'     
     if file_test(outfile) then begin
        print,'Already mapped ',c
        continue
     endif

; compute relative distances to cluster center
     if numpart_total[0] gt 0l and numpart_total[4] gt 0l then begin
        x=[reform(pos_dm[0,*])-pos_fof[0,ifof],reform(pos_gas[0,*])-pos_fof[0,ifof],reform(pos_star[0,*])-pos_fof[0,ifof]]
        y=[reform(pos_dm[1,*])-pos_fof[1,ifof],reform(pos_gas[1,*])-pos_fof[1,ifof],reform(pos_star[1,*])-pos_fof[1,ifof]]
        z=[reform(pos_dm[2,*])-pos_fof[2,ifof],reform(pos_gas[2,*])-pos_fof[2,ifof],reform(pos_star[2,*])-pos_fof[2,ifof]]
        mtmp=dblarr(n_elements(reform(pos_dm[0,*])))
        mtmp[*]=mdm
        
        m=[mtmp,mgas,mstar]
        m*=1d10/h
     endif
     if numpart_total[0] gt 0l  and numpart_total[4] eq 0l then begin
        x=[reform(pos_dm[0,*])-pos_fof[0,ifof],reform(pos_gas[0,*])-pos_fof[0,ifof]]
        y=[reform(pos_dm[1,*])-pos_fof[1,ifof],reform(pos_gas[1,*])-pos_fof[1,ifof]]
        z=[reform(pos_dm[2,*])-pos_fof[2,ifof],reform(pos_gas[2,*])-pos_fof[2,ifof]]
        
        mtmp=dblarr(n_elements(reform(pos_dm[0,*])))
        mtmp[*]=mdm

        m=[mtmp,mgas]
        m*=1d10/h
     endif
     if numpart_total[0] eq 0l  and numpart_total[4] eq 0l then begin
        x=reform(pos_dm[0,*])-pos_fof[0,ifof]
        y=reform(pos_dm[1,*])-pos_fof[1,ifof]
        z=reform(pos_dm[2,*])-pos_fof[2,ifof]
        m=dblarr(n_elements(reform(pos_dm[0,*])))
        m[*]=mdm*1d10/h
     endif

     m = getPositions( numpart_total, pos_fof, $
                          pos_dm, pos_gas, pos_star, $
                          mdm, mgas, mstar, ifof, h, $
                          projection,$
                       x=x, y=y, z=z)
; wrap
     periodic_wrap,boxsize,x,y,z
      
; switch to physical Mpc
     x*=aexp/h
     y*=aexp/h
     z*=aexp/h 
     
     index_p=where(abs(x) le fov/2d0 AND abs(y) le fov/2d0 AND abs(z) le dz/2d0,ndm)
     x=x[index_p]
     y=y[index_p]
     z=z[index_p]
     m=m[index_p]
     undefine,index_p
     
; shift coordinate system for image making purposes
     x+=fov/2d0
     y+=fov/2d0
     z+=dz/2d0
     
     
     pos=fltarr(3,ndm)
     pos[0,*]=x
     pos[1,*]=y
     pos[2,*]=z
     hsml=fltarr(ndm)
     Hmax=fov
     ResultW = fltarr(npix,npix)
     ResultQ = fltarr(npix,npix)
     Object_File = "HsmlAndProject.so"
     S = CALL_EXTERNAL(Object_File, "findHsmlAndProject",long(ndm),float(pos),float(hsml),float(m),float(m),float(0.0),float(fov),float(0.0),float(fov),float(0.0),float(dz),long(npix),long(npix),long(24),long(0),long(1),long(2),float(Hmax),fov,float(ResultW),float(ResultQ))     
     image=rotate(ResultW,4)/dpix/dpix
     undefine,pos
     undefine,Hmax
     undefine,ResultW
     undefine,ResultQ
;      imdisp,alog10(image),margin=0,range=[alog10(max(image))-3d0,alog10(max(image))],/negative

     writefits,outfile,image
     print,c,total(image)
     
     x*=npix/fov
     y*=npix/fov
     image=tsc(m,x,npix,y,npix,/NO_MESSAGE)/dpix/dpix

     print,c,ifof,alog10(M200[ifof]),r200[ifof],format='(2I10,2F10.3)'
     

     c+=1l
     
     
  endfor

  endfor

endfor


end
